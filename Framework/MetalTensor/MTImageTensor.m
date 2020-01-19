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
}

- (instancetype)initWithImage:(UIImage *)image normalized:(BOOL)normalized {
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
        
        float16_t *float16 = malloc(width * height * 3 * sizeof(float16_t));
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int index = row*width+col;
                float16[index*3] = data[index*4+1];     // B
                float16[index*3+1] = data[index*4+2];   // G
                float16[index*3+2] = data[index*4+3];   // R
            }
        }
        free(data);
        if (normalized) {
            for (int i = 0; i < width*height*3; i++) {
                float16[i] /= 255.0f;
            }
        }
        [self loadData:float16 length:width * height * 3 * sizeof(float16_t)];
        free(float16);
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
        int size = ProductOfDataShapeDepth4Divisible(self.shape);
        _result = malloc(size * sizeof(float16_t));
        [_mpsImage toFloat16Array:_result];
        ConvertF16ToTensorFlowLayout1(_result, self.shape);
    }
}

- (void)printResult {
    
    int row = self.shape->row;
    int column = self.shape->column;
    int depth = self.shape->depth;
    
    printf("\nTensor: %dx%dx%d", row, column, depth);
    for (int i = 0; i < row; i++) {
        printf("\nrow:%d", i);
        printf("\n(");
        for (int j = 0; j < column; j++) {
            printf("\n  col:%d", j);
            printf("\n  (");
            for (int c = 0; c < depth; c++) {
                printf("%f", _result[(i*column+j)*depth+c]);
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
    
    assert(x < column);
    assert(y < row);
    
    printf("\nTensor: %dx%dx%d", row, column, depth);
    printf("\nPixel(%d, %d):", x, y);
    printf("\n    (");
    depth = MAX(depth, 4);
    for (int c = 0; c < depth; c++) {
        printf("%f", _result[(y*column+x)*depth+c]);
        if (c < depth-1) {
            printf(", ");
        }
    }
    printf(")   \n");
}

#endif

@end
