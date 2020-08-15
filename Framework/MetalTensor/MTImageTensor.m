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
    
    float16_t *_buffer;
}


- (instancetype)initWithImage:(UIImage *)image normalized:(BOOL)normalized {
    return [self initWithImage:image normalized:normalized frameBuffer:NO];
}

- (instancetype)initWithImage:(UIImage *)image normalized:(BOOL)normalized frameBuffer:(BOOL)frameBuffer {
    return [self initWithImage:image normalized:normalized frameBuffer:frameBuffer rgb:NO];
}

- (instancetype)initWithImage:(UIImage *)image normalized:(BOOL)normalized frameBuffer:(BOOL)frameBuffer rgb:(BOOL)rgb {
    int width = (int)image.size.width;
    int height = (int)image.size.height;
    DataShape shape = DataShapeMake(height, width, 3);
    if (self = [super initWithShape:&shape]) {
        self.referenceCountingEnable = NO;
        
        MPSImageDescriptor *descriptor = ImageDescriptor(&shape, MPSDataTypeFloat16);
        descriptor.storageMode = MTLStorageModeShared;
        _mpsImage = [[MPSImage alloc] initWithDevice:[MetalDevice sharedMTLDevice] imageDescriptor:descriptor];
        
        static CGColorSpaceRef colorSpace = NULL;
        if (colorSpace == NULL) {
            colorSpace = CGColorSpaceCreateDeviceRGB();
        }
        Byte *data = malloc(width * height * 4 * sizeof(Byte));
        if (!data) {
            goto SKIP;
        }
        // We have got a RGBA data layout format in litte endian, which equals to ABGR in big endian.
        CGContextRef context = CGBitmapContextCreate(data, width, height, 8, width*4, colorSpace, kCGImageByteOrder32Little|kCGImageAlphaPremultipliedLast);
        CGContextDrawImage(context, CGContextGetClipBoundingBox(context), image.CGImage);
        CGContextRelease(context);
        
        _buffer = malloc(Product(&shape) * sizeof(float16_t));
        
        if (!_buffer) {
            goto SKIP;
        }
        
        if (rgb) {
            // RGB
            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    int index = row*width+col;
                    _buffer[index*3] = data[index*4+3];
                    _buffer[index*3+1] = data[index*4+2];
                    _buffer[index*3+2] = data[index*4+1];
                }
            }
        }
        else {
            // BGR
            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    int index = row*width+col;
                    _buffer[index*3] = data[index*4+1];
                    _buffer[index*3+1] = data[index*4+2];
                    _buffer[index*3+2] = data[index*4+3];  
                }
            }
        }
        
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
    
SKIP:
        if (data) {
            free(data);
            data = NULL;
        }
        
    }
    return self;
}


- (instancetype)initWithShape:(DataShape *)shape {
    if (self = [super initWithShape:shape]) {
        self.referenceCountingEnable = NO;
        
        MPSImageDescriptor *descriptor = ImageDescriptor(shape, MPSDataTypeFloat16);
        descriptor.storageMode = MTLStorageModeShared;
        _mpsImage = [[MPSImage alloc] initWithDevice:[MetalDevice sharedMTLDevice] imageDescriptor:descriptor];
    }
    return self;
}

- (instancetype)initWithShape:(DataShape *)shape dataType:(MPSDataType)dataType {
    if (self = [super initWithShape:shape dataType:dataType numberOfImage:1]) {
        self.referenceCountingEnable = NO;
        
        MPSImageDescriptor *descriptor = ImageDescriptor(shape, dataType);
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
#if DEBUG
    if (_result) {
        free(_result);
        _result = NULL;
    }
#endif
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

- (void)newContentOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    NSAssert(NO, @"It is invalid to call on command buffer for 'MTImageTensor', which is a global variance.");
    _mpsImage = nil;
    [super newContentOnCommandBuffer:commandBuffer];
}

- (MPSImage *)content {
    return _mpsImage;
}

#if DEBUG

- (void)dump {
    
    if (_result == NULL) {
        double interval = CACurrentMediaTime();
        int size = ProductDepth4Divisible(self.shape);
        _result = malloc(size * sizeof(float16_t));
        [_mpsImage toFloat16Array:_result];
        ConvertF16ToTensorFlowLayout1(_result, self.shape);
        printf("\ndump tensor: %dx%dx%d, %f ms.", self.shape->row, self.shape->column, self.shape->depth, (CACurrentMediaTime()-interval)*1000.0f);
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
            if (buffer_depth >= 4) {
                printf("\n  col:%d", j);
                printf("\n  (");
            }
            for (int c = 0; c < buffer_depth; c++) {
                printf("%e", _result[(i*column+j)*buffer_depth+c]);
                if (c < depth-1) {
                    printf(", ");
                }
            }
            if (buffer_depth >= 4) {
                printf(")");
            }
            if (j < column-1) {
                printf(", ");
            }
        }
        printf("\n  )");
    }
    printf("\n");
}

- (void)printResultCHW {
    
    int row = self.shape->row;
    int column = self.shape->column;
    int depth = self.shape->depth;
    
    printf("\nTensor: %dx%dx%d\n\n", row, column, depth);
    for (int ch = 0; ch < depth; ch++) {
        for (int r = 0; r < row; r ++) {
            for (int col = 0; col < column; col++) {
                printf("%0.1f, ", _result[(r*column+col)*depth+ch]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

- (void)printResultHWC {
    
    int row = self.shape->row;
    int column = self.shape->column;
    int depth = self.shape->depth;
    
    printf("\nTensor: %dx%dx%d\n\n", row, column, depth);
    for (int h = 0; h < row; h++) {
        for (int w = 0; w < column; w ++) {
            for (int c = 0; c < depth; c++) {
                printf("%0.1f, ", _result[(h*column+w)*depth+c]);
            }
            printf("\n");
        }
        printf("\n");
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

- (void)innerChannelsProduct {
    int row = self.shape->row;
    int column = self.shape->column;
    int depth = self.shape->depth;
    
    printf("\ninner product begin: %f", CACurrentMediaTime());
    float32_t *buffer = malloc(Product(self.shape) * sizeof(float32_t));
    
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < column; c++) {
            for (int d = 0; d < depth; d++) {
                buffer[(r*column+c)*depth+d] = _result[(r*column+c)*depth+d] * _result[(column+c)*depth+d];
            }
        }
    }

    printf("\ninner product end: %f", CACurrentMediaTime());
    printf("\nTensor: %dx%dx%d", row, column, depth);
    int y0 = 0;
    int y1 = 4;
    int x0 = 0;
    int x1 = 4;
    
    int n_slice = (depth+3)/4;
    int n_component = _mpsImage.numberOfComponents;
    int buffer_depth = n_slice * n_component;
    
    printf("\nPixels[%d:%d, %d:%d, :]", y0, y1, x0, x1);
    for (int y = y0; y < y1; y ++) {
        printf("\n");
        for (int x = x0; x < x1; x ++) {
            printf("(%d, %d) (", y, x);
            for (int c = 0; c < buffer_depth; c++) {
                printf("%e", buffer[(y*column+x)*buffer_depth+c]);
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
    free(buffer);
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
