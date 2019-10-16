//
//  MetalTensorNeuronLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/5.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorNeuronLayer.h"
#import "MITemporaryImageCache.h"

@interface MetalTensorNeuronLayer() {
    MPSCNNNeuron *_neuron;
}

@end

@implementation MetalTensorNeuronLayer

- (void)compile:(id<MTLDevice>)device {

    [super compile:device];
    
    MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:_neuronType.neuron
                                                                                         a:_neuronType.a
                                                                                         b:_neuronType.b
                                                                                         c:_neuronType.c];
    _neuron = [[MPSCNNNeuron alloc] initWithDevice:device neuronDescriptor:neuronDesc];
}

- (void)setNeuronType:(NeuronType)neuronType {
    _neuronType = neuronType;
    
    if (_device) {
        MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:_neuronType.neuron
                                                                                             a:_neuronType.a
                                                                                             b:_neuronType.b
                                                                                             c:_neuronType.c];
        _neuron = [[MPSCNNNeuron alloc] initWithDevice:_device neuronDescriptor:neuronDesc];
    }
}

- (void)setInputImage:(MITemporaryImage *)newInputImage atIndex:(NSInteger)imageIndex {
    NSAssert(DataShapesTheSame(newInputImage.shape, &_inputShapes[0]), @"Invalid input tensor shape.");
    [super setInputImage:newInputImage atIndex:imageIndex];
}

- (void)processTensorWithCommandBuffer:(id<MTLCommandBuffer>)cmdBuf {
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);

    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:cmdBuf];
    [_outputTempImage newTemporaryImageForCommandBuffer:cmdBuf];
    [_neuron encodeToCommandBuffer:cmdBuf
                       sourceImage:_inputs[@(0)].image
                  destinationImage:_outputTempImage.image];
    [self removeCachedImages];
    
    [self notifyTargetsAboutNewTempImage:cmdBuf];
}

@end
