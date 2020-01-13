//
//  MetalTensorInputLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/5.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorInputLayer.h"

@implementation MetalTensorInputLayer {
    MPSCNNNeuron *_neuron;
}

#pragma mark - override
- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
    [self buildupTensors];
    
    if (_needBackward) {
        MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeNone];
        _neuron = [[MPSCNNNeuron alloc] initWithDevice:_device neuronDescriptor:neuronDesc];
    }
}

#pragma mark - private
- (void)buildupTensors {
    
    if (_device) {
        _outputImage = [[MTImageTensor alloc] initWithShape:&_outputShape];
        
        if (_needBackward) {
            _gradientImage = [[MTImageTensor alloc] initWithShape:&_outputShape];
        }
    }
}

#pragma mark - public
- (void)inputTexture:(id<MTLTexture>)bgraU8Texture {
    DB_TRACE(-_verbose+2, "\n\n\n%s <-- (%ld, %ld, 4)", self.labelUTF8, bgraU8Texture.width, bgraU8Texture.height);
//    [self debugInputTexture:bgraU8Texture];
    _outputImage.mpsImage = [[MPSImage alloc] initWithTexture:bgraU8Texture featureChannels:3];
}

#pragma mark - MTTensorForward Delegate
- (void)setInputShape:(DataShape *)dataShape atIndex:(NSInteger)imageIndex {
    [super setInputShape:dataShape atIndex:imageIndex];
    [self buildupTensors];
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {

    for (ForwardTarget currentTarget in _targets) {
        NSInteger indexOfObject = [_targets indexOfObject:currentTarget];
        NSInteger imageIndex = [_targetIndices[indexOfObject] integerValue];
        
        [currentTarget setImage:_outputImage atIndex:imageIndex];
        
        DB_TRACE(-_verbose+1, "\n%s ---%s---> %s(%ld)", self.labelUTF8,
                 NSStringFromDataShape(_outputImage.shape).UTF8String,
                 [currentTarget description].UTF8String, imageIndex);
        
        [currentTarget imageReadyOnCommandBuffer:commandBuffer atIndex:imageIndex];
    }
}

#pragma mark - MTTensorBackward Delegate
- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    [_neuron encodeToCommandBuffer:commandBuffer
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
