//
//  MetalTensorNeuronLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/5.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorNeuronLayer.h"
#import <MetalImage/MetalDevice.h>
#import "MITemporaryImageCache.h"

@interface MetalTensorNeuronLayer() {
    MPSCNNNeuron *_neuron;
}

@end

@implementation MetalTensorNeuronLayer

- (instancetype)initWithDataShape:(DataShape *)dataShape neuronType:(NeuronType)neuronType {
    if (self = [super initWithInputShape:dataShape outputShape:dataShape]) {
        _neuronType = neuronType;
        MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:neuronType.neuron a:neuronType.a b:neuronType.b c:neuronType.c];
        _neuron = [[MPSCNNNeuron alloc] initWithDevice:[MetalDevice sharedMTLDevice] neuronDescriptor:neuronDesc];

        DB_TRACE(-_verbose+2, "\n%s init --> %s", self.labelUTF8, NSStringFromDataShape(dataShape).UTF8String);
    }
    return self;
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
