//
//  MISeparableConvolutionLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/6.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MISeparableConvolutionLayer.h"
#include "numpy.h"

@interface MISeparableConvolutionLayer() {
    
    MIConvolutionLayer *_depthwise;
    MIConvolutionLayer *_project;
}

@end

@implementation MISeparableConvolutionLayer

- (void)dealloc {
    if (_kernels) {
        free(_kernels);
        _kernels = NULL;
    }
    
    if (_neurons) {
        free(_neurons);
        _neurons = NULL;
    }
}

#pragma mark - override
- (void)initialize {
    _kernels = malloc(2 * sizeof(KernelShape));
    _neurons = malloc(2 * sizeof(NeuronType));
    _padding = MTPaddingMode_tfsame;
}

- (void)compile:(id<MTLDevice>)device {

    [super compile:device];
    
    _depthwise = [[MIConvolutionLayer alloc] initWithInputShape:&_inputShapes[0]];
    _depthwise.kernel = _kernels[0];
    _depthwise.neuron = _neurons[0];
    _depthwise.depthWise = YES;
    _depthwise.padding = _padding;
    _depthwise.offset = _offset;
    _depthwise.dataSource = _dataSourceDepthWise;
    [_depthwise compile:device];
    
    _project = [[MIConvolutionLayer alloc] initWithInputShape:_depthwise.outputShapeRef];
    _project.kernel = _kernels[1];
    _project.neuron = _neurons[1];
    _project.depthWise = NO;
    _project.padding = MTPaddingMode_tfsame;
    _project.dataSource = _dataSourceProject;
    [_project compile:device];
    
    [_depthwise addTarget:_project];
    
    _outputShape = _project.outputShape;
    
    [self setLabel:_label];
}

- (void)addTarget:(ForwardTarget)newTarget {
    NSAssert(_project, @"The separable layer is not compiled yet.");
    [_project addTarget:newTarget];
}

- (void)addTarget:(ForwardTarget)newTarget atIndex:(NSInteger)imageIndex {
    NSAssert(_project, @"The separable layer is not compiled yet.");
    [_project addTarget:newTarget atIndex:imageIndex];
}

- (void)removeTarget:(ForwardTarget)targetToRemove {
    NSAssert(_project, @"The separable layer is not compiled yet.");
    [_project removeTarget:targetToRemove];
}

- (void)removeAllTargets {
    NSAssert(_project, @"The separable layer is not compiled yet.");
    [_project removeAllTargets];
}

- (void)setLabel:(NSString *)label {
    [super setLabel:label];
    
    [_depthwise setLabel:[NSString stringWithFormat:@"%@_depthwise", label]];
    [_project setLabel:[NSString stringWithFormat:@"%@_project", label]];
}

#pragma mark - public
- (void)setPadding:(MTPaddingMode)padding {
    _padding = padding;
    [_depthwise setPadding:_padding];
}

- (void)setOffset:(MTLInt2)offset {
    _offset = offset;
    [_depthwise setOffset:offset];
}

- (void)setDataSourceDepthWise:(MICNNKernelDataSource *)dataSourceDepthWise {
    _dataSourceDepthWise = dataSourceDepthWise;
    _depthwise.dataSource = dataSourceDepthWise;
}

- (void)setDataSourceProject:(MICNNKernelDataSource *)dataSourceProject {
    _dataSourceProject = dataSourceProject;
    _project.dataSource = dataSourceProject;
}

- (MIConvolutionLayer *)depthwiseComponent {
    return _depthwise;
}

- (MIConvolutionLayer *)projectComponent {
    return _project;
}

#ifdef DEBUG
- (void)setVerbose:(int)verbose {
    [super setVerbose:verbose];
    
    [_depthwise setVerbose:verbose];
    [_project setVerbose:verbose];
}
#endif

#pragma mark - MTTensorForward Delegate
- (void)setInputShape:(DataShape *)dataShape atIndex:(NSInteger)imageIndex {
    [_depthwise setInputShape:dataShape atIndex:imageIndex];
}

- (void)setImage:(MetalTensor)newImage atIndex:(NSInteger)imageIndex {
    [_depthwise setImage:newImage atIndex:imageIndex];
}

- (void)imageReadyOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer atIndex:(NSInteger)imageIndex {
    [_depthwise imageReadyOnCommandBuffer:commandBuffer atIndex:imageIndex];
}

- (void)reserveImageIndex:(NSInteger)index {
    [_depthwise reserveImageIndex:index];
}

- (void)releaseImageIndex:(NSInteger)index {
    [_depthwise releaseImageIndex:index];
}
#pragma mark - Management of the weights

- (BOOL)didLoadWeights {
    return [_depthwise didLoadWeights] && [_project didLoadWeights];
}

- (void)loadWeights {
    [_depthwise loadWeights];
    [_project loadWeights];
}

- (void)loadWeightsList:(NSArray<NSString *> *)weightsList rangeList:(NSRange *)rangeList {
    [_depthwise loadWeights:weightsList[0] range:&rangeList[0]];
    [_project loadWeights:weightsList[1] range:&rangeList[1]];
}

MISeparableConvolutionLayer *MakeSeparableConvolutionLayer(DataShape *input,
                                                           KernelShape *kernels,
                                                           NeuronType *neurons,
                                                           MICNNKernelDataSource *depthWiseData,
                                                           MICNNKernelDataSource *pointWiseData,
                                                           NSString * __nullable name)
{
    MISeparableConvolutionLayer *separableLayer = [[MISeparableConvolutionLayer alloc] initWithInputShape:input];
    separableLayer.dataSourceProject = pointWiseData;
    separableLayer.dataSourceDepthWise = depthWiseData;
    npmemcpy(separableLayer.kernels, kernels, 2 * sizeof(KernelShape));
    npmemcpy(separableLayer.neurons, neurons, 2 * sizeof(NeuronType));
    separableLayer.label = name;
    return separableLayer;
}

@end
