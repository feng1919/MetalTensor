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

- (void)setStopGradient:(BOOL)stopGradient {
    NSAssert(NO, @"The arithmetic layer does not support stop gradient.");
}

#pragma mark - override

- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
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
        [self.blit setDestinationFeatureChannelOffset:_offsets[i]];
        [self.blit encodeToCommandBuffer:commandBuffer sourceImage:tensor.content destinationImage:_image.content];
    }
    
    if (!_needBackward) {
        [self removeCachedImages];
    }
}

#pragma mark - MTTensorBackward Delegate

- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    DB_TRACE(-_verbose+2, "\n%s backward encoding...", self.labelUTF8);
    
    [self.blit setDestinationFeatureChannelOffset:0];
    for (int i = 0; i < _numOfImages; i++) {
        MetalTensor tensor = _inputImages[@(i)];
        BackwardTarget backwardTarget = tensor.source;
        NSAssert(backwardTarget, @"Invalid backward target[%d]...", i);
        
        [self.blit setSourceFeatureChannelOffset:_offsets[i]];
        [self.blit encodeToCommandBuffer:commandBuffer sourceImage:_gradient.content destinationImage:tensor.content];
        [backwardTarget setGradient:tensor forwardTarget:self];
        [tensor unlock];
        [backwardTarget gradientReadyOnCommandBuffer:commandBuffer forwardTarget:self];
    }
    
    [_inputImages removeAllObjects];
}

@end
