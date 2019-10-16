//
//  MISoftMaxLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MISoftMaxLayer.h"
#import <MetalImage/MetalDevice.h>
#import "MITemporaryImageCache.h"

@interface MISoftMaxLayer() {
    MPSCNNSoftMax *_softMax;
}

@end

@implementation MISoftMaxLayer

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    _softMax = [[MPSCNNSoftMax alloc] initWithDevice:device];
}

- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);

    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:commandBuffer];
    [_outputTempImage newTemporaryImageForCommandBuffer:commandBuffer];
    [_softMax encodeToCommandBuffer:commandBuffer
                       sourceImage:_inputs[@(0)].image
                  destinationImage:_outputTempImage.image];
    
    [self removeCachedImages];
    [self notifyTargetsAboutNewTempImage:commandBuffer];
}

@end
