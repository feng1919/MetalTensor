//
//  MTImageTensor.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/25.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MTImageTensor.h"
#import <MetalImage/MetalDevice.h>

@implementation MTImageTensor {
    
#if DEBUG
    float16_t *_result;
#endif
    
    float16_t *_buffer;
}


- (instancetype)initWithImage:(UIImage *)image normalized:(BOOL)normalized {
    return [self initWithImage:image normalized:normalized frameBuffer:NO];
}

- (instancetype)initWithImage:(UIImage *)image normalized:(BOOL)normalized frameBuffer:(BOOL)frameBuffer {
    int width = (int)image.size.width;
    int height = (int)image.size.height;
    DataShape shape = DataShapeMake(height, width, 3);
    if (self = [super initWithShape:&shape]) {
        self.referenceCountingEnable = NO;
        
        MPSImageDescriptor *descriptor = ImageDescriptor(&shape, TensorDataFormatFloat16);
        descriptor.storageMode = MTLStorageModeShared;
        _mpsImage = [[MPSImage alloc] initWithDevice:[MetalDevice sharedMTLDevice] imageDescriptor:descriptor];
        
        static CGColorSpaceRef colorSpace = NULL;
        if (colorSpace == NULL) {
            colorSpace = CGColorSpaceCreateDeviceRGB();
        }
        Byte *data = malloc(width * height * 4 * sizeof(Byte));
        // We have got a RGBA data layout format in litte endian, which equals to ABGR in big endian.
        CGContextRef context = CGBitmapContextCreate(data, width, height, 8, width*4, colorSpace, kCGImageByteOrder32Little|kCGImageAlphaPremultipliedLast);
        CGContextDrawImage(context, CGContextGetClipBoundingBox(context), image.CGImage);
        CGContextRelease(context);
        
        _buffer = malloc(Product(&shape) * sizeof(float16_t));
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int index = row*width+col;
                _buffer[index*3] = data[index*4+1];     // B
                _buffer[index*3+1] = data[index*4+2];   // G
                _buffer[index*3+2] = data[index*4+3];   // R
            }
        }
        free(data);
        if (normalized) {
            for (int i = 0; i < width*height*3; i++) {
                _buffer[i] /= 255.0f;
            }
        }
        [self loadData:_buffer length:Product(&shape) * sizeof(float16_t)];
        
        if (frameBuffer == NO) {
            free(_buffer);
            _buffer = NULL;
        }
    }
    return self;
}

- (instancetype)initWithShape:(DataShape *)shape {
    if (self = [super initWithShape:shape]) {
        self.referenceCountingEnable = NO;
        
        MPSImageDescriptor *descriptor = ImageDescriptor(shape, TensorDataFormatFloat16);
        descriptor.storageMode = MTLStorageModeShared;
        _mpsImage = [[MPSImage alloc] initWithDevice:[MetalDevice sharedMTLDevice] imageDescriptor:descriptor];
    }
    return self;
}

- (instancetype)initWithContent:(MPSImage *)mpsImage {
    NSAssert(![mpsImage isKindOfClass:[MPSTemporaryImage class]], @"Invalid content");
    
    DataShape shape = DataShapeMake((int)mpsImage.height, (int)mpsImage.width, (int)mpsImage.featureChannels);
    if (self = [super initWithShape:&shape]) {
        self.referenceCountingEnable = NO;
        
        _mpsImage = mpsImage;
    }
    return self;
}

- (void)dealloc {
    if (_buffer) {
        free(_buffer);
        _buffer = NULL;
    }
}

- (float16_t *)buffer {
    return _buffer;
}

- (void)loadBuffer {
    NSAssert(_buffer, @"There is no buffer to load.");
    if (_buffer) {
        [self loadData:_buffer length:Product(self.shape)*sizeof(float16_t)];
    }
}

- (void)loadData:(float16_t *)data length:(NSInteger)length {
    [_mpsImage writeBytes:data dataLayout:MPSDataLayoutHeightxWidthxFeatureChannels imageIndex:0];
}

- (MPSTemporaryImage *)newContentOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    NSAssert(NO, @"It is invalid to call on command buffer for 'MTImageTensor', which is a global variance.");
    _mpsImage = nil;
    return [super newContentOnCommandBuffer:commandBuffer];
}

- (MPSImage *)content {
    return _mpsImage;
}

#if DEBUG

- (void)dump {
    
    if (_result == NULL) {
        int size = ProductDepth4Divisible(self.shape);
        _result = malloc(size * sizeof(float16_t));
        [_mpsImage toFloat16Array:_result];
        ConvertF16ToTensorFlowLayout1(_result, self.shape);
    }
}

- (void)printResult {
    
    int row = self.shape->row;
    int column = self.shape->column;
    int depth = self.shape->depth;
    
    int n_slice = (depth+3)/4;
    int n_component = _mpsImage.numberOfComponents;
    int buffer_depth = n_slice * n_component;
    
    printf("\nTensor: %dx%dx%d", row, column, depth);
    for (int i = 0; i < row; i++) {
        printf("\nrow:%d", i);
        printf("\n(");
        for (int j = 0; j < column; j++) {
            printf("\n  col:%d", j);
            printf("\n  (");
            for (int c = 0; c < buffer_depth; c++) {
                printf("%f", _result[(i*column+j)*buffer_depth+c]);
                if (c < depth-1) {
                    printf(", ");
                }
            }
            printf("),");
        }
        printf("\n  )");
    }
    printf("\n");
}

- (void)printPixelAtX:(int)x Y:(int)y {
    
    int row = self.shape->row;
    int column = self.shape->column;
    int depth = self.shape->depth;
    
    int n_slice = (depth+3)/4;
    int n_component = _mpsImage.numberOfComponents;
    int buffer_depth = n_slice * n_component;
    
    assert(x < column);
    assert(y < row);
    
    float sum = 0.0f;
    
    printf("\nTensor: %dx%dx%d", row, column, depth);
    printf("\nPixel(%d, %d):", x, y);
    printf("\n    (");
    for (int c = 0; c < buffer_depth; c++) {
        printf("%e", _result[(y*column+x)*buffer_depth+c]);
        sum += _result[(y*column+x)*buffer_depth+c];
        if (c < buffer_depth-1) {
            printf(", ");
        }
        if ((c+1)%4 == 0) {
            printf("\n");
        }
    }
    printf(")   \n");
    
    printf("Sum:%e\n", sum);
}

- (void)printPixelsFromX:(int)x0 toX:(int)x1 fromY:(int)y0 toY:(int)y1 {
    
    int row = self.shape->row;
    int column = self.shape->column;
    int depth = self.shape->depth;
    
    int n_slice = (depth+3)/4;
    int n_component = _mpsImage.numberOfComponents;
    int buffer_depth = n_slice * n_component;
    
    assert(x0 >= 0 && x0 < x1 && x1 <= column);
    assert(y0 >= 0 && y0 < y1 && y1 <= row);
    
    printf("\nTensor: %dx%dx%d", row, column, depth);
    printf("\nPixels[%d:%d, %d:%d, :]", y0, y1, x0, x1);
    for (int y = y0; y < y1; y ++) {
        printf("\n");
        for (int x = x0; x < x1; x ++) {
            printf("(%d, %d) (", y, x);
            for (int c = 0; c < buffer_depth; c++) {
                printf("%e", _result[(y*column+x)*buffer_depth+c]);
                if (c < buffer_depth-1) {
                    printf(", ");
                    if ((c+1)%4 == 0) {
                        printf("\n");
                    }
                }
            }
            printf(")\n");
        }
    }
}

- (void)checkNan {
    int row = self.shape->row;
    int column = self.shape->column;
    int depth = self.shape->depth;
    
    unsigned long long count = row * column * depth;
    unsigned long long nan_count = 0;
    for (unsigned long long i = 0; i < count; i++) {
        if (isnan(_result[i])) {
            nan_count ++;
        }
    }
    printf("\nnan count: %lld", nan_count);
}

- (void)checkInf {
    int row = self.shape->row;
    int column = self.shape->column;
    int depth = self.shape->depth;
    
    unsigned long long count = row * column * depth;
    unsigned long long inf_count = 0;
    for (unsigned long long i = 0; i < count; i++) {
        if (isinf(_result[i])) {
            inf_count ++;
        }
    }
    printf("\ninf count: %lld", inf_count);
}

#endif

@end
