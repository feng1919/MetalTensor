//
//  MIBatchNormalizationLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIBatchNormalizationLayer.h"
#import <MetalImage/MetalDevice.h>
#import "MITemporaryImageCache.h"

@interface MIBatchNormalizationLayer() {
    MPSCNNBatchNormalization *bn;
}

@end

@implementation MIBatchNormalizationLayer

- (instancetype)init {
    if (self = [super init]) {
        _edgeMode = MPSImageEdgeModeZero;
        _epsilon = 0.001;
    }
    return self;
}

- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape
                      kernelDataSource:(id<MPSCNNBatchNormalizationDataSource>)dataSource {
    if (self = [super initWithInputShape:inputShape outputShape:outputShape]) {
        _dataSource = dataSource;
        _edgeMode = MPSImageEdgeModeZero;
        _epsilon = 0.001;
    }
    return self;
}

- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    NSAssert(_dataSource != nil, @"The weights has not been set.");
    if (bn == nil) {
        bn = [[MPSCNNBatchNormalization alloc] initWithDevice:[MetalDevice sharedMTLDevice]
                                                   dataSource:_dataSource];
        bn.epsilon = _epsilon;
        bn.edgeMode = _edgeMode;
    }
    
    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:commandBuffer];
    [_outputTempImage newTemporaryImageForCommandBuffer:commandBuffer];
    [bn encodeToCommandBuffer:commandBuffer
                  sourceImage:_inputs[@(0)].image
             destinationImage:_outputTempImage.image];
    
    [self removeCachedImages];
    
    [self notifyTargetsAboutNewTempImage:commandBuffer];
    
}

@end
