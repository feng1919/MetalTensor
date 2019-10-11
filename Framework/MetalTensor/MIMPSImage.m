//
//  MIMPSImage.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/25.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIMPSImage.h"
#import <MetalImage/MetalDevice.h>

@implementation MIMPSImage

- (instancetype)initWithImage:(UIImage *)image normalized:(BOOL)normalized {
    int width = (int)image.size.width;
    int height = (int)image.size.height;
    DataShape shape = DataShapeMake(height, width, 3);
    if (self = [super initWithShape:&shape]) {
        self.referenceCountingEnable = NO;
        
        MPSImageDescriptor *descriptor = ImageDescriptor(&shape);
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
        [_mpsImage writeBytes:float16 dataLayout:MPSDataLayoutHeightxWidthxFeatureChannels imageIndex:0];
        free(float16);
    }
    return self;
}

- (instancetype)initWithShape:(DataShape *)shape {
    if (self = [super initWithShape:shape]) {
        self.referenceCountingEnable = NO;
        
        MPSImageDescriptor *descriptor = ImageDescriptor(shape);
        descriptor.storageMode = MTLStorageModeShared;
        _mpsImage = [[MPSImage alloc] initWithDevice:[MetalDevice sharedMTLDevice] imageDescriptor:descriptor];
    }
    return self;
}

- (MPSTemporaryImage *)newTemporaryImageForCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    NSAssert(NO, @"It is invalid to call on command buffer for 'MIMPSImage', which is a global variance.");
    _mpsImage = nil;
    return [super newTemporaryImageForCommandBuffer:commandBuffer];
}

- (MPSTemporaryImage *)image {
    return (MPSTemporaryImage *)_mpsImage;
}

@end
