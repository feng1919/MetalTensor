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

- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
    _offsets = malloc(_numOfImages * sizeof(int));
    _dataShape = ConcatenateShapes(_inputShapes, _numOfImages, _offsets, true);
    _tensorShape = ConcatenateShapes(_inputShapes, _numOfImages, NULL, false);

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

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    DB_TRACE(-_verbose+2, "\n%s forward encoding...", self.labelUTF8);
    
    _image = [[MTTensorCache sharedCache] fetchTensorWithShape:&_dataShape source:self commandBuffer:commandBuffer];
    [_image newContentOnCommandBuffer:commandBuffer];
    
    for (int i = 0; i < _numOfImages; i++) {
        MetalTensor tensor = _inputImages[@(i)];
        [_neuron setDestinationFeatureChannelOffset:_offsets[i]];
        [_neuron encodeToCommandBuffer:commandBuffer sourceImage:tensor.content destinationImage:_image.content];
    }
    
    if (!_needBackward) {
        [self removeCachedImages];
    }
}

- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    DB_TRACE(-_verbose+2, "\n%s backward encoding...", self.labelUTF8);
    
    [_neuron setDestinationFeatureChannelOffset:0];
    for (int i = 0; i < _numOfImages; i++) {
        MetalTensor tensor = _inputImages[@(i)];
        [_neuron setSourceFeatureChannelOffset:_offsets[i]];
        [_neuron encodeToCommandBuffer:commandBuffer sourceImage:_gradient.content destinationImage:tensor.content];
        [tensor.source setGradient:tensor forwardTarget:self];
        [tensor unlock];
        [tensor.source gradientReadyFromForwardTarget:self onCommandBuffer:commandBuffer];
    }
    
    [_inputImages removeAllObjects];
}

@end
