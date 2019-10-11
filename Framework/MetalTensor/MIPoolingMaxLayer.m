//
//  MIPoolingMaxLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIPoolingMaxLayer.h"
#import <MetalImage/MetalDevice.h>
#import "MITemporaryImageCache.h"

@interface MIPoolingMaxLayer() {
    MPSCNNPoolingMax *_poolingMax;
}

@end

@implementation MIPoolingMaxLayer

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
        _kernelWidth = 2;
        _kernelHeight = 2;
        _strideInPixelsX = 2;
        _strideInPixelsY = 2;
        _offset.x = 1;
        _offset.y = 1;
    }
    return self;
}

- (void)setOffsetWithX:(NSInteger)x Y:(NSInteger)y Z:(NSInteger)z {
    _offset.x = x;
    _offset.y = y;
    _offset.z = z;
    _poolingMax.offset = _offset;
}

- (void)processTensorWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);
    
    if (_poolingMax == nil) {
        _poolingMax = [[MPSCNNPoolingMax alloc] initWithDevice:[MetalDevice sharedMTLDevice]
                                                  kernelWidth:_kernelWidth
                                                 kernelHeight:_kernelHeight
                                              strideInPixelsX:_strideInPixelsX
                                              strideInPixelsY:_strideInPixelsY];
        _poolingMax.offset = _offset;
        _poolingMax.edgeMode = MPSImageEdgeModeClamp;
    }
    
    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:commandBuffer];
    [_outputTempImage newTemporaryImageForCommandBuffer:commandBuffer];
    [_poolingMax encodeToCommandBuffer:commandBuffer
                          sourceImage:_inputs[@(0)].image
                     destinationImage:_outputTempImage.image];
    
    [self removeCachedImages];
    
    [self notifyTargetsAboutNewTempImage:commandBuffer];
}

@end
