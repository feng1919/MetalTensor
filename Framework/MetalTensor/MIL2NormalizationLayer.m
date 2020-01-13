//
//  MIL2NormalizationLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/9.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIL2NormalizationLayer.h"
#import "MTTensorCache.h"

@interface MIL2NormalizationLayer() {
    
    MPSCNNPoolingL2Norm *_l2Normalization;
    MPSCNNPoolingL2NormGradient *_l2NormaliationGradientOp;
}

@end

@implementation MIL2NormalizationLayer

#pragma mark - override
- (void)initialize {
    _kernel = KernelShapeMake(2, 2, 1, 1, 2);
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    NSParameterAssert(_kernel.stride > 0);
    
    _l2Normalization = [[MPSCNNPoolingL2Norm alloc] initWithDevice:_device
                                                       kernelWidth:_kernel.column
                                                      kernelHeight:_kernel.row
                                                   strideInPixelsX:_kernel.stride
                                                   strideInPixelsY:_kernel.stride];
    _l2Normalization.offset = _offset;
    
    if (_needBackward) {
        _l2NormaliationGradientOp = [[MPSCNNPoolingL2NormGradient alloc] initWithDevice:_device
                                                                            kernelWidth:_kernel.column
                                                                           kernelHeight:_kernel.row
                                                                        strideInPixelsX:_kernel.stride
                                                                        strideInPixelsY:_kernel.stride];
    }
    
    _operation = _l2Normalization;
    _gradientOp = _l2NormaliationGradientOp;
}

- (void)updateOutputShape {
    if (_device) {
        _outputShape.column = pooling_output_length(_inputShapes[0].column, _kernel.stride);
        _outputShape.row = pooling_output_length(_inputShapes[0].row, _kernel.stride);
        _outputShape.depth = _inputShapes[0].depth;
    }
}

#pragma mark - public

- (void)setOffset:(MPSOffset)offset {
    _offset = offset;
    _l2Normalization.offset = _offset;
}

- (void)setKernel:(KernelShape)kernel {
    
    NSParameterAssert(kernel.stride > 0);
    
    _kernel = kernel;
}

@end
