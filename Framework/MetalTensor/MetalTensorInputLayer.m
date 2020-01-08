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

- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
    _outputImage = [[MTImageTensor alloc] initWithShape:&_dataShape];
    [_outputImage setReferenceCountingEnable:NO];
    
    MPSImageDescriptor *descriptor = ImageDescriptor(&_dataShape);
    descriptor.storageMode = MTLStorageModeShared;
    _outputImage.mpsImage = [[MPSImage alloc] initWithDevice:_device imageDescriptor:descriptor];

    if (_needBackward) {
        MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeNone];
        _neuron = [[MPSCNNNeuron alloc] initWithDevice:device neuronDescriptor:neuronDesc];
        
        MPSImageDescriptor *desc = ImageDescriptor(&_dataShape);
        desc.storageMode = MTLStorageModeShared;
        _gradientImage.mpsImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];
        
    }
}

- (void)inputTexture:(id<MTLTexture>)bgraU8Texture {
    DB_TRACE(-_verbose+2, "\n\n\n%s <-- (%ld, %ld, 4)", self.labelUTF8, bgraU8Texture.width, bgraU8Texture.height);
    _outputImage.mpsImage = [[MPSImage alloc] initWithTexture:bgraU8Texture featureChannels:3];
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {

    for (ForwardTarget currentTarget in _targets) {
        NSInteger indexOfObject = [_targets indexOfObject:currentTarget];
        NSInteger imageIndex = [_targetIndices[indexOfObject] integerValue];
        
        [currentTarget setImage:_outputImage atIndex:imageIndex];
        
        DB_TRACE(-_verbose+1, "\n%s ---%s---> %s(%ld)", self.labelUTF8,
                 NSStringFromDataShape(_outputImage.shape).UTF8String,
                 [currentTarget description].UTF8String, imageIndex);
        
        [currentTarget imageReadyAtIndex:imageIndex onCommandBuffer:commandBuffer];
    }
}

- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    [_neuron encodeToCommandBuffer:commandBuffer
                       sourceImage:_gradient.content
                  destinationImage:_gradientImage.mpsImage];
}

@end
