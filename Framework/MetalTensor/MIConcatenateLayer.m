//
//  MIConcatenateLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/9.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIConcatenateLayer.h"
#import "MTTensorCache.h"
#import "MIArithmeticLayer.h"
#include "numpy.h"

@interface MIConcatenateLayer() {
    MPSCNNNeuron *_neuron;
    int *_offsets;
}

@end

@implementation MIConcatenateLayer

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

#pragma mark - override

- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
    MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeNone];
    _neuron = [[MPSCNNNeuron alloc] initWithDevice:device neuronDescriptor:neuronDesc];
}

- (void)updateOutputShape {
    
    if (_device) {
        if (_offsets == NULL) {
            _offsets = malloc(_numOfImages * sizeof(int));
        }

        _outputShape = ConcatenateShapes(_inputShapes, _numOfImages, _offsets, true);
        _tensorShape = ConcatenateShapes(_inputShapes, _numOfImages, NULL, false);
    }
}

#pragma mark - MTTensorForward Delegate

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    DB_TRACE(-_verbose+2, "\n%s forward encoding...", self.labelUTF8);
    
    _image = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape commandBuffer:commandBuffer];
    _image.source = self;
    
    for (int i = 0; i < _numOfImages; i++) {
        MetalTensor tensor = _inputImages[@(i)];
        [_neuron setDestinationFeatureChannelOffset:_offsets[i]];
        [_neuron encodeToCommandBuffer:commandBuffer sourceImage:tensor.content destinationImage:_image.content];
    }
    
    if (!_needBackward) {
        [self removeCachedImages];
    }
}

#pragma mark - MTTensorBackward Delegate

- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    DB_TRACE(-_verbose+2, "\n%s backward encoding...", self.labelUTF8);
    
    [_neuron setDestinationFeatureChannelOffset:0];
    for (int i = 0; i < _numOfImages; i++) {
        MetalTensor tensor = _inputImages[@(i)];
        [_neuron setSourceFeatureChannelOffset:_offsets[i]];
        [_neuron encodeToCommandBuffer:commandBuffer sourceImage:_gradient.content destinationImage:tensor.content];
        [tensor.source setGradient:tensor forwardTarget:self];
        [tensor unlock];
        [tensor.source gradientReadyOnCommandBuffer:commandBuffer forwardTarget:self];
    }
    
    [_inputImages removeAllObjects];
}

@end
