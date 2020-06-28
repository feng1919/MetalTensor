//
//  MetalTensorInputLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/5.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorInputLayer.h"

@implementation MetalTensorInputLayer

#pragma mark - override
- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
    [self buildupTensors];
}

#pragma mark - private
- (void)buildupTensors {
    
    if (_device) {
        _outputImage = [[MTImageTensor alloc] initWithShape:&_outputShape dataType:_dataType];
        
        if (_needBackward) {
            _gradientImage = [[MTImageTensor alloc] initWithShape:&_outputShape dataType:_dataType];
        }
    }
}

#pragma mark - public
- (void)inputTexture:(id<MTLTexture>)bgraU8Texture {
    DB_TRACE(-_verbose+2, "\n\n\n%s <-- (%ld, %ld, 4)", self.labelUTF8, bgraU8Texture.width, bgraU8Texture.height);
//    [self debugInputTexture:bgraU8Texture];
    _outputImage.mpsImage = [[MPSImage alloc] initWithTexture:bgraU8Texture featureChannels:3];
    _outputImage.source = self;
    _image = _outputImage;
}

- (void)inputTensor:(MetalTensor)tensor {
    _image = tensor;
    _image.source = self;
    [_image lock];
}

#pragma mark - MTTensorForward Delegate
- (void)setInputShape:(DataShape *)dataShape atIndex:(NSInteger)imageIndex {
    [super setInputShape:dataShape atIndex:imageIndex];
    [self buildupTensors];
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
}

- (void)notifyTargetsAboutNewImageOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
#if DEBUG
    if (self.dumpResult) {
        [self saveTensor:_image onCommandBuffer:commandBuffer];
    }
#endif
    [super notifyTargetsAboutNewImageOnCommandBuffer:commandBuffer];
}

//- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
//
//    for (ForwardTarget currentTarget in _targets) {
//        NSInteger indexOfObject = [_targets indexOfObject:currentTarget];
//        NSInteger imageIndex = [_targetIndices[indexOfObject] integerValue];
//
//        [currentTarget setImage:_image atIndex:imageIndex];
//
//        DB_TRACE(-_verbose+1, "\n%s ---%s---> %s(%ld)", self.labelUTF8,
//                 NSStringFromDataShape(_outputImage.shape).UTF8String,
//                 [currentTarget description].UTF8String, imageIndex);
//    }
//
//    [_image unlock];
//}

#pragma mark - MTTensorBackward Delegate
- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    [self.blit encodeToCommandBuffer:commandBuffer
                           sourceImage:_gradient.content
                      destinationImage:_gradientImage.mpsImage];
    
    [self removeGradient];
}

#pragma mark - DEBUG
- (void)debugInputTexture:(id<MTLTexture>)texture {
    
    CGSize imageSize = CGSizeMake([texture width], [texture height]);
    size_t imageByteCount = imageSize.width * imageSize.height * 4;
    Byte *imageBytes = malloc(imageByteCount);
    NSUInteger bytesPerRow = imageSize.width * 4;
    MTLRegion region = MTLRegionMake2D(0, 0, imageSize.width, imageSize.height);
    [texture getBytes:imageBytes bytesPerRow:bytesPerRow fromRegion:region mipmapLevel:0];
    
    for (int i = 0; i < 10; i++) {
        printf("\n%d, %d, %d", imageBytes[i*4], imageBytes[i*4+1], imageBytes[i*4+2]);
    }
    printf("\n");
    
    free(imageBytes);
}

@end
