//
//  MIBatchNormalizationLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIBatchNormalizationLayer.h"
#import "MTTensorCache.h"

@interface MIBatchNormalizationLayer() {
    MPSCNNBatchNormalization *_bn;
    MPSCNNBatchNormalizationGradient *_bnGradientOp;
}

@end

@implementation MIBatchNormalizationLayer

- (void)initialize {
    _edgeMode = MPSImageEdgeModeZero;
    _epsilon = 0.001;
}

- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
    [self updateComputing];
}

- (void)setDataSource:(id<MPSCNNBatchNormalizationDataSource>)dataSource {
    _dataSource = dataSource;
    [self updateComputing];
}

- (void)updateComputing {
    
    if (_device && _dataSource) {
        _bn = [[MPSCNNBatchNormalization alloc] initWithDevice:_device dataSource:_dataSource];
        _bn.epsilon = _epsilon;
        _bn.edgeMode = _edgeMode;
        
        if (_needBackward) {
            _bnGradientOp = [[MPSCNNBatchNormalizationGradient alloc] initWithDevice:_device fusedNeuronDescriptor:nil];
        }
        
        _operation = _bn;
        _gradientOp = _bnGradientOp;
    }
}

@end
