//
//  MISoftMaxLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MISoftMaxLayer.h"
#import "MTTensorCache.h"

@interface MISoftMaxLayer() {
    MPSCNNSoftMax *_softMax;
    MPSCNNSoftMaxGradient *_softmaxGradientOp;
}

@end

@implementation MISoftMaxLayer

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    _softMax = [[MPSCNNSoftMax alloc] initWithDevice:device];
    if (_needBackward) {
        _softmaxGradientOp = [[MPSCNNSoftMaxGradient alloc] initWithDevice:device];
    }
    _operation = _softMax;
    _gradientOp = _softmaxGradientOp;
}


@end
