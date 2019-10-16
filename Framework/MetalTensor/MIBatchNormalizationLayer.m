//
//  MIBatchNormalizationLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIBatchNormalizationLayer.h"
#import "MITemporaryImageCache.h"

@interface MIBatchNormalizationLayer() {
    MPSCNNBatchNormalization *_bn;
}

@end

@implementation MIBatchNormalizationLayer

- (void)initialize {
    _edgeMode = MPSImageEdgeModeZero;
    _epsilon = 0.001;
}

- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
    if (_dataSource) {
        _bn = [[MPSCNNBatchNormalization alloc] initWithDevice:_device dataSource:_dataSource];
        _bn.epsilon = _epsilon;
        _bn.edgeMode = _edgeMode;
    }
}

- (void)setDataSource:(id<MPSCNNBatchNormalizationDataSource>)dataSource {
    _dataSource = dataSource;
    
    if (_device) {
        _bn = [[MPSCNNBatchNormalization alloc] initWithDevice:_device dataSource:_dataSource];
        _bn.epsilon = _epsilon;
        _bn.edgeMode = _edgeMode;
    }
}

- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    NSAssert(_dataSource != nil, @"The weights has not been set.");
    
    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:commandBuffer];
    [_outputTempImage newTemporaryImageForCommandBuffer:commandBuffer];
    [_bn encodeToCommandBuffer:commandBuffer
                   sourceImage:_inputs[@(0)].image
              destinationImage:_outputTempImage.image];
    
    [self removeCachedImages];
    
    [self notifyTargetsAboutNewTempImage:commandBuffer];
    
}

@end
