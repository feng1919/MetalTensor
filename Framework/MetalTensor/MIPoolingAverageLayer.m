//
//  MIPoolingAverageLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/24.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIPoolingAverageLayer.h"
#import "MTTensorCache.h"

@interface MIPoolingAverageLayer() {
    MPSCNNPoolingAverage *_pooling;
    MPSCNNPoolingAverageGradient *_poolingAverageGradientOp;
}

@end

@implementation MIPoolingAverageLayer

- (void)initialize {
    _kernel = KernelShapeMake(2, 2, 1, 1, 2);
}

- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
    NSParameterAssert(_kernel.stride > 0);
    _dataShape.column = conv_output_length(_inputShapes[0].column, _kernel.column, _kernel.stride, MTPaddingMode_tfsame);
    _dataShape.row = conv_output_length(_inputShapes[0].row, _kernel.row, _kernel.stride, MTPaddingMode_tfsame);
    _dataShape.depth = _inputShapes[0].depth;
    
    _pooling = [[MPSCNNPoolingAverage alloc] initWithDevice:_device
                                                kernelWidth:_kernel.column
                                               kernelHeight:_kernel.row
                                            strideInPixelsX:_kernel.stride
                                            strideInPixelsY:_kernel.stride];
    _pooling.offset = _offset;
    _pooling.edgeMode = MPSImageEdgeModeClamp;
    
    if (_needBackward) {
        _poolingAverageGradientOp = [[MPSCNNPoolingAverageGradient alloc] initWithDevice:_device
                                                                             kernelWidth:_kernel.column
                                                                            kernelHeight:_kernel.row
                                                                         strideInPixelsX:_kernel.stride
                                                                         strideInPixelsY:_kernel.stride];
    }
    
    _operation = _pooling;
    _gradientOp = _poolingAverageGradientOp;
}

- (void)setOffset:(MPSOffset)offset {
    _offset = offset;
    _pooling.offset = _offset;
}

- (void)setKernel:(KernelShape)kernel {
    
    NSParameterAssert(kernel.stride > 0);
    
    _kernel = kernel;
}

@end
