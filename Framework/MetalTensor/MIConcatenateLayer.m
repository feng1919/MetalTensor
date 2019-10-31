//
//  MIConcatenateLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/9.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIConcatenateLayer.h"
#import "MITemporaryImageCache.h"
#import "MIArithmeticLayer.h"
#include "numpy.h"

@interface MIConcatenateLayer() {
    MPSCNNNeuron *_neuron;
    int *_offsets;
}

@end

@implementation MIConcatenateLayer

- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
    _offsets = malloc(_numOfInputs * sizeof(int));
    _outputShape = ConcatenateShapes(_inputShapes, _numOfInputs, _offsets, true);
    _tensorShape = ConcatenateShapes(_inputShapes, _numOfInputs, NULL, false);

    _neuron = [[MPSCNNNeuron alloc] initWithDevice:device
                                  neuronDescriptor:[MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeNone]];
    
}

- (void)dealloc {
    
    if (_offsets) {
        free(_offsets);
        _offsets = NULL;
    }
}

- (DataShape *)tensorShape {
    return &_tensorShape;
}

- (int *)channelOffsets {
    return _offsets;
}

#pragma mark - MetalTensorInput Delegate

- (void)processTensorWithCommandBuffer:(id<MTLCommandBuffer>)cmdBuf {
    
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);
    
    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:cmdBuf];
    [_outputTempImage newTemporaryImageForCommandBuffer:cmdBuf];
    
    for (int i = 0; i < _numOfInputs; i++) {
        MITemporaryImage *tensor = _inputs[@(i)];
        [_neuron setDestinationFeatureChannelOffset:_offsets[i]];
        [_neuron encodeToCommandBuffer:cmdBuf sourceImage:tensor.image destinationImage:_outputTempImage.image];
    }
    
    [self removeCachedImages];
    
    [self notifyTargetsAboutNewTempImage:cmdBuf];
}

@end
