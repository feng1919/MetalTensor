//
//  MIConvolutionLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIConvolutionLayer.h"
#import "MITemporaryImageCache.h"
#import "MIDataSource.h"

@interface MIConvolutionLayer() {
    
    MPSCNNConvolution *_convolution;
}

@end

@implementation MIConvolutionLayer

- (void)initialize {
    _edgeMode = MPSImageEdgeModeZero;
    _depthWise = NO;
    _neuron.neuron = MPSCNNNeuronTypeNone;
    _neuron.a = 0.0f;
    _neuron.b = 0.0f;
    _padding = MTPaddingMode_tfsame;
    _offset.x = 0;
    _offset.y = 0;
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    _outputShape.column = conv_output_length(_inputShapes[0].column, _kernel.column, _kernel.stride, _padding);
    _outputShape.row = conv_output_length(_inputShapes[0].row, _kernel.row, _kernel.stride, _padding);
    _outputShape.depth = _depthWise?_inputShapes[0].depth:_kernel.filters;
    
    if (_dataSource) {
        _convolution = [[MPSCNNConvolution alloc] initWithDevice:_device weights:_dataSource];
        [_convolution setEdgeMode:_edgeMode];
        [self setOffset:_offset];
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

    MPSOffset mpsOffset;
    mpsOffset.x = _offset.x;
    mpsOffset.y = _offset.y;
    mpsOffset.z = 0;
    [_convolution setOffset:mpsOffset];
}

- (void)setDataSource:(MICNNKernelDataSource *)dataSource {
    
    NSParameterAssert(dataSource);
    
    _dataSource = dataSource;
    
    DB_TRACE(-_verbose+1, "\n%s data source --> %s", self.labelUTF8, [dataSource description].UTF8String);
    
    if (_device) {
        _convolution = [[MPSCNNConvolution alloc] initWithDevice:_device weights:_dataSource];
        [_convolution setEdgeMode:_edgeMode];
        [self setOffset:_offset];
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
    self.dataSource = MakeDataSource2(weights, &_kernel, &_neuron, _depthWise, &range[0]);;
    [self.dataSource load];
}

MIConvolutionLayer *MakeConvolutionLayer(NSString *module_name,
                                         KernelShape *k,
                                         NeuronType *n,
                                         MTPaddingMode padding,
                                         DataShape *input)
{
    MICNNKernelDataSource *data_cnn = MakeDataSource(module_name, k, n);
    MIConvolutionLayer *module = [[MIConvolutionLayer alloc] initWithInputShape:input];
    module.kernel = *k;
    module.neuron = *n;
    module.padding = padding;
    module.dataSource = data_cnn;
    module.label = module_name;
    return module;
}

@end
