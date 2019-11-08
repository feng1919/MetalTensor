//
//  MITransposeConvolutionLayer.m
//  MetalTensorDemo
//
//  Created by Feng Stone on 2019/9/24.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MITransposeConvolutionLayer.h"
#import "MITemporaryImageCache.h"
#import "MIDataSource.h"

@interface MITransposeConvolutionLayer() {
    
    MPSCNNConvolutionTranspose *_convolution;
}

@end

@implementation MITransposeConvolutionLayer

- (void)initialize {

    _edgeMode = MPSImageEdgeModeZero;
    _neuron.neuron = MPSCNNNeuronTypeNone;
    _neuron.a = 0.0f;
    _neuron.b = 0.0f;
    _neuron.c = 0.0f;
    _depthWise = NO;
    _offset.x = 0;
    _offset.y = 0;
    
    DB_TRACE(-_verbose+2, "\n%s init %s", self.labelUTF8, NSStringFromDataShape(&_inputShapes[0]).UTF8String);
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    _outputShape.column = trans_conv_output_length(_inputShapes[0].column, _kernel.column, _kernel.stride, _padding);
    _outputShape.row = trans_conv_output_length(_inputShapes[0].row, _kernel.row, _kernel.stride, _padding);
    _outputShape.depth = _depthWise?_inputShapes[0].depth:_kernel.filters;
    
    if (_dataSource) {
        
        _convolution = [[MPSCNNConvolutionTranspose alloc] initWithDevice:device weights:_dataSource];
        _convolution.edgeMode = _edgeMode;
        [_convolution setKernelOffsetX:_offset.x];
        [_convolution setKernelOffsetY:_offset.y];
        
    }
}

- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    NSAssert(_dataSource != nil, @"The weights has not been set.");
    
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);
    
    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:commandBuffer];
    [_outputTempImage newTemporaryImageForCommandBuffer:commandBuffer];
    [_convolution encodeToCommandBuffer:commandBuffer
                           sourceImage:_inputs[@(0)].image
                      destinationImage:_outputTempImage.image];
    
    [self removeCachedImages];
    
    [self notifyTargetsAboutNewTempImage:commandBuffer];
}

- (void)setEdgeMode:(MPSImageEdgeMode)edgeMode {
    _edgeMode = edgeMode;
    [_convolution setEdgeMode:_edgeMode];
}

- (void)setOffset:(MTLInt2)offset {
    _offset = offset;
    [_convolution setKernelOffsetX:offset.x];
    [_convolution setKernelOffsetX:offset.y];
}

- (void)setDataSource:(id<MPSCNNConvolutionDataSource>)dataSource {
    NSParameterAssert(dataSource);
    _dataSource = dataSource;
    
    DB_TRACE(-_verbose+1, "\n%s data source --> %s", self.labelUTF8, [dataSource description].UTF8String);
    
    if (_device) {
        
        _convolution = [[MPSCNNConvolutionTranspose alloc] initWithDevice:_device weights:_dataSource];
        _convolution.edgeMode = _edgeMode;
        [_convolution setKernelOffsetX:_offset.x];
        [_convolution setKernelOffsetY:_offset.y];
    }
}

#pragma mark - Management of the weights

- (BOOL)didLoadWeights {
    return self.dataSource != nil;
}

- (void)loadWeights {
    NSAssert(_dataSource, @"The weights data source object is not initialized yet.");
    [self.dataSource load];
}

- (void)loadWeights:(NSData *)weights {
    self.dataSource = [[MICNNKernelDataSource alloc] initWithData:weights kernel:&_kernel neuron:&_neuron depthWise:_depthWise];
    [self.dataSource load];
}

- (void)loadWeights:(NSString *)weights range:(NSRange *)range {
    self.dataSource = MakeDataSource2(weights, &_kernel, &_neuron, _depthWise, &range[0]);
    [self.dataSource load];
}

@end
