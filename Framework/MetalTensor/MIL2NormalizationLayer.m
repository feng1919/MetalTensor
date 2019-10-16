//
//  MIL2NormalizationLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/9.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIL2NormalizationLayer.h"
#import <MetalImage/MetalDevice.h>
#import "MITemporaryImageCache.h"

@interface MIL2NormalizationLayer() {
    
    MPSCNNPoolingL2Norm *_l2Normalization;
}

@end

@implementation MIL2NormalizationLayer

- (void)initialize {
    _kernel = KernelShapeMake(2, 2, 1, 1, 2);
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    NSParameterAssert(_kernel.stride > 0);
    
    _outputShape.column = (_inputShapes[0].column + _kernel.stride - 1) / _kernel.stride;
    _outputShape.row = (_inputShapes[0].row + _kernel.stride - 1) / _kernel.stride;
    _outputShape.depth = _inputShapes[0].depth;
    
    _l2Normalization = [[MPSCNNPoolingL2Norm alloc] initWithDevice:_device
                                                       kernelWidth:_kernel.column
                                                      kernelHeight:_kernel.row
                                                   strideInPixelsX:_kernel.stride
                                                   strideInPixelsY:_kernel.stride];
    _l2Normalization.offset = _offset;
}

- (void)setOffset:(MPSOffset)offset {
    _offset = offset;
    _l2Normalization.offset = _offset;
}

- (void)setKernel:(KernelShape)kernel {
    
    NSParameterAssert(kernel.stride > 0);
    
    _kernel = kernel;
}

- (void)processTensorWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);
    
    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:commandBuffer];
    [_outputTempImage newTemporaryImageForCommandBuffer:commandBuffer];
    [_l2Normalization encodeToCommandBuffer:commandBuffer
                               sourceImage:_inputs[@(0)].image
                          destinationImage:_outputTempImage.image];
    
    [self removeCachedImages];
    
    [self notifyTargetsAboutNewTempImage:commandBuffer];
}

@end
