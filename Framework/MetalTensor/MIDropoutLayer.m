//
//  MIDropoutLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/24.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIDropoutLayer.h"
#import "MITemporaryImageCache.h"

@interface MIDropoutLayer() {
    
    MPSCNNDropout *_dropout;
}

@end

@implementation MIDropoutLayer

- (void)initialize {
    _keepProbability = 0.999f;
}

- (void)setKeepProbability:(float)keepProbability {
    _keepProbability = keepProbability;
    DB_TRACE(-_verbose+1, "\n%s.keepProbability --> %f", self.labelUTF8, keepProbability);
    
    if (_device) {
        _dropout = [[MPSCNNDropout alloc] initWithDevice:_device
                                         keepProbability:_keepProbability
                                                    seed:0
                                      maskStrideInPixels:MTLSizeMake(1, 1, 1)];
    }
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];

    _dropout = [[MPSCNNDropout alloc] initWithDevice:device
                                     keepProbability:_keepProbability
                                                seed:0
                                  maskStrideInPixels:MTLSizeMake(1, 1, 1)];
}

- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);
    
    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:commandBuffer];
    [_outputTempImage newTemporaryImageForCommandBuffer:commandBuffer];
    [_dropout encodeToCommandBuffer:commandBuffer
                       sourceImage:_inputs[@(0)].image
                  destinationImage:_outputTempImage.image];
    
    [self removeCachedImages];
    
    [self notifyTargetsAboutNewTempImage:commandBuffer];
}

@end
