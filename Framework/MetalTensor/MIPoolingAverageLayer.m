//
//  MIPoolingAverageLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/24.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIPoolingAverageLayer.h"
#import <MetalImage/MetalDevice.h>
#import "MITemporaryImageCache.h"

@interface MIPoolingAverageLayer() {
    MPSCNNPoolingAverage *_pooling;
}

@end

@implementation MIPoolingAverageLayer

- (void)initialize {
    _kernel = KernelShapeMake(2, 2, 1, 1, 2);
}

- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
    NSParameterAssert(_kernel.stride > 0);
    _outputShape.column = conv_output_length(_inputShapes[0].column, _kernel.column, _kernel.stride, MTPaddingMode_tfsame);
    _outputShape.row = conv_output_length(_inputShapes[0].row, _kernel.row, _kernel.stride, MTPaddingMode_tfsame);
    _outputShape.depth = _inputShapes[0].depth;
    
    _pooling = [[MPSCNNPoolingAverage alloc] initWithDevice:_device
                                                kernelWidth:_kernel.column
                                               kernelHeight:_kernel.row
                                            strideInPixelsX:_kernel.stride
                                            strideInPixelsY:_kernel.stride];
    _pooling.offset = _offset;
    _pooling.edgeMode = MPSImageEdgeModeClamp;
}

- (void)setOffset:(MPSOffset)offset {
    _offset = offset;
    _pooling.offset = _offset;
}

- (void)setKernel:(KernelShape)kernel {
    
    NSParameterAssert(kernel.stride > 0);
    
    _kernel = kernel;
}

- (void)processTensorWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);
    
    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:commandBuffer];
    [_outputTempImage newTemporaryImageForCommandBuffer:commandBuffer];
    [_pooling encodeToCommandBuffer:commandBuffer
                        sourceImage:_inputs[@(0)].image
                   destinationImage:_outputTempImage.image];
    
    [self removeCachedImages];
    
    [self notifyTargetsAboutNewTempImage:commandBuffer];
}

@end
