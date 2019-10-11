//
//  MIL2NormalizationLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/9.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIL2NormalizationLayer.h"
#import <MetalImage/MetalDevice.h>
#import "MITemporaryImageCache.h"

@interface MIL2NormalizationLayer() {
    
    MPSCNNPoolingL2Norm *_l2Normalization;
}

@end

@implementation MIL2NormalizationLayer

- (instancetype)init {
    if (self = [super init]) {
        _kernelWidth = 2;
        _kernelHeight = 2;
        _strideInPixelsX = 2;
        _strideInPixelsY = 2;
        _offset.x = 1;
        _offset.y = 1;
    }
    return self;
}

- (instancetype)initWithOutputShape:(DataShape *)outputShape {
    if (self = [super initWithOutputShape:outputShape]) {
        _kernelWidth = 2;
        _kernelHeight = 2;
        _strideInPixelsX = 2;
        _strideInPixelsY = 2;
        _offset.x = 1;
        _offset.y = 1;
    }
    return self;
}

- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape {
    if (self = [super initWithInputShape:inputShape outputShape:outputShape]) {
        _kernelWidth = inputShape->column;
        _kernelHeight = inputShape->row;
        _strideInPixelsX = 1;
        _strideInPixelsY = 1;
        _offset.x = (inputShape->column-1)>>1;
        _offset.y = (inputShape->row-1)>>1;
    }
    return self;
}

- (void)processTensorWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);
    
    if (_l2Normalization == nil) {
        _l2Normalization = [[MPSCNNPoolingL2Norm alloc] initWithDevice:[MetalDevice sharedMTLDevice]
                                                          kernelWidth:_kernelWidth
                                                         kernelHeight:_kernelHeight
                                                      strideInPixelsX:_strideInPixelsX
                                                      strideInPixelsY:_strideInPixelsY];
        _l2Normalization.offset = _offset;
    }
    
    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:commandBuffer];
    [_outputTempImage newTemporaryImageForCommandBuffer:commandBuffer];
    [_l2Normalization encodeToCommandBuffer:commandBuffer
                               sourceImage:_inputs[@(0)].image
                          destinationImage:_outputTempImage.image];
    
    [self removeCachedImages];
    
    [self notifyTargetsAboutNewTempImage:commandBuffer];
}

@end
