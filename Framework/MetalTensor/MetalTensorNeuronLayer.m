//
//  MetalTensorNeuronLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/5.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorNeuronLayer.h"
#import "MTTensorCache.h"

@interface MetalTensorNeuronLayer() {
    MPSCNNNeuron *_neuron;
    MPSCNNNeuronGradient *_neuronGradientOp;
}

- (void)updateComputation;

@end

@implementation MetalTensorNeuronLayer

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    [self updateComputation];
}

- (void)setNeuronType:(NeuronType)neuronType {
    _neuronType = neuronType;
    [self updateComputation];
}

- (void)updateComputation {
    if (_device) {
        MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:_neuronType.neuron
                                                                                             a:_neuronType.a
                                                                                             b:_neuronType.b
                                                                                             c:_neuronType.c];
        _neuron = [[MPSCNNNeuron alloc] initWithDevice:_device neuronDescriptor:neuronDesc];

        if (_needBackward) {
            _neuronGradientOp = [[MPSCNNNeuronGradient alloc] initWithDevice:_device neuronDescriptor:neuronDesc];
        }
        
        _operation = _neuron;
        _gradientOp = _neuronGradientOp;
    }
}

- (void)setImage:(MetalTensor)newImage atIndex:(NSInteger)imageIndex {
    NSAssert(DataShapesTheSame(newImage.shape, &_inputShapes[0]), @"Invalid input tensor shape.");
    [super setImage:newImage atIndex:imageIndex];
}

@end
