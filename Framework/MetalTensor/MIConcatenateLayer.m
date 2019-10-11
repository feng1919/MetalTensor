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
#import <MetalImage/MetalDevice.h>
#include "numpy.h"

@interface MIConcatenateLayer() {
    MPSCNNNeuron *_neuron;
    int *_offsets;
}

@end

@implementation MIConcatenateLayer

- (instancetype)initWithInputShapes1:(DataShape * _Nonnull *)inputShapes size:(int)size {
    
    _offsets = malloc(size * sizeof(int));
    _outputShape = ConcatenateShapes1(inputShapes, size, _offsets, true);
    if (self = [super initWithInputShapes1:inputShapes size:size outputShape:&_outputShape]) {
        
        _tensorShape = ConcatenateShapes1(inputShapes, size, NULL, false);
        
        _neuron = [[MPSCNNNeuron alloc] initWithDevice:[MetalDevice sharedMTLDevice] neuronDescriptor:[MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeNone]];
        
#if DEBUG
        NSMutableString *string = [NSMutableString string];
        for (int i = 0; i < size; i ++) {
            [string appendString:NSStringFromDataShape(inputShapes[i])];
            if (i != size-1) {
                [string appendString:@", "];
            }
        }
        
        DB_TRACE(-_verbose+2, "\n%s init1 [%s] --> %s", self.labelUTF8, string.UTF8String, NSStringFromDataShape(&_outputShape).UTF8String);
#endif
        
    }
    return self;
}

- (instancetype)initWithInputShapes:(DataShape *_Nonnull)inputShapes size:(int)size {
    
    _offsets = malloc(size * sizeof(int));
    _outputShape = ConcatenateShapes(inputShapes, size, _offsets, true);
    if (self = [super initWithInputShapes:inputShapes size:size outputShape:&_outputShape]) {
        
        _tensorShape = ConcatenateShapes(inputShapes, size, NULL, false);
        
        _neuron = [[MPSCNNNeuron alloc] initWithDevice:[MetalDevice sharedMTLDevice] neuronDescriptor:[MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeNone]];
        
#if DEBUG
        NSMutableString *string = [NSMutableString string];
        for (int i = 0; i < size; i ++) {
            [string appendString:NSStringFromDataShape(&inputShapes[i])];
            if (i != size-1) {
                [string appendString:@", "];
            }
        }
        
        DB_TRACE(-_verbose+2, "\n%s init [%s] --> %s", self.labelUTF8, string.UTF8String, NSStringFromDataShape(&_outputShape).UTF8String);
#endif
        
    }
    return self;
}

- (void)dealloc {
    
    free(_offsets);
    _offsets = NULL;
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
