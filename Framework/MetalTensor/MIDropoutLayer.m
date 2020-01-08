//
//  MIDropoutLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/24.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIDropoutLayer.h"
#import "MTTensorCache.h"

@interface MIDropoutLayer() {
    
    MPSCNNDropout *_dropout;
    MPSCNNDropoutGradient *_dropoutGradientOp;
}

@end

@implementation MIDropoutLayer

- (void)initialize {
    _keepProbability = 0.999f;
}

- (void)setKeepProbability:(float)keepProbability {
    _keepProbability = keepProbability;
    DB_TRACE(-_verbose+1, "\n%s.keepProbability --> %f", self.labelUTF8, keepProbability);
    
    [self updateComputing];
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];

    [self updateComputing];
}

- (void)updateComputing {
    
    if (_device) {
        _dropout = [[MPSCNNDropout alloc] initWithDevice:_device
                                         keepProbability:_keepProbability
                                                    seed:0
                                      maskStrideInPixels:MTLSizeMake(1, 1, 1)];
        
        if (_needBackward) {
            _dropoutGradientOp = [[MPSCNNDropoutGradient alloc] initWithDevice:_device
                                                               keepProbability:_keepProbability
                                                                          seed:0
                                                            maskStrideInPixels:MTLSizeMake(1, 1, 1)];
        }
        
        _operation = _dropout;
        _gradientOp = _dropoutGradientOp;
    }
}

@end
