//
//  MIInvertedResidualModule.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/21.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIInvertedResidualModule.h"
#import "MIArithmeticLayer.h"
#import "MIConvolutionLayer.h"
#import "MIDataSource.h"
#include "numpy.h"

@interface MIInvertedResidualModule() {
    
    MIConvolutionLayer *_convExpand;
    MIConvolutionLayer *_convDepthWise;
    MIConvolutionLayer *_convProject;
    MIArithmeticLayer *_addition;
    MetalTensorNode *_lastNode;
}

@end

@implementation MIInvertedResidualModule

- (void)initialize {
    _kernels = malloc(3 * sizeof(KernelShape));
    _neurons = malloc(3 * sizeof(NeuronType));
    _offset = MTLInt2Make(0, 0);
}

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

- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
    _convExpand = [[MIConvolutionLayer alloc] initWithInputShape:&_inputShapes[0]];
    _convExpand.padding = MTPaddingMode_tfsame;
    _convExpand.depthWise = NO;
    _convExpand.kernel = _kernels[0];
    _convExpand.neuron = _neurons[0];
    [_convExpand compile:device];
    
    _convDepthWise = [[MIConvolutionLayer alloc] initWithInputShape:_convExpand.outputShapeRef];
    _convDepthWise.padding = MTPaddingMode_tfsame;
    _convDepthWise.depthWise = YES;
    _convDepthWise.offset = _offset;
    _convDepthWise.kernel = _kernels[1];
    _convDepthWise.neuron = _neurons[1];
    [_convDepthWise compile:device];
    
    _convProject = [[MIConvolutionLayer alloc] initWithInputShape:_convDepthWise.outputShapeRef];
    _convProject.padding = MTPaddingMode_tfsame;
    _convProject.depthWise = NO;
    _convProject.kernel = _kernels[2];
    _convProject.neuron = _neurons[2];
    [_convProject compile:device];
    
    _outputShape = _convProject.outputShape;

    [_convExpand addTarget:_convDepthWise];
    [_convDepthWise addTarget:_convProject];
    
    if (DataShapesTheSame(&_inputShapes[0], &_outputShape)) {
        _addition = [MIArithmeticLayer arithmeticLayerWithDataShape:&_outputShape];
        _addition.arithmeticType = @"addition";
        [_addition compile:device];
        
        [_convProject addTarget:_addition atTempImageIndex:1];
        _lastNode = _addition;
    }
    else {
        _lastNode = _convProject;
    }
    
    [self setLabel:_label];
    
    DB_TRACE(-_verbose+2, "\n%s compile %s --> %s --> %s --> %s", self.labelUTF8,
             NSStringFromDataShape(&_inputShapes[0]).UTF8String,
             NSStringFromDataShape(_convExpand.outputShapeRef).UTF8String,
             NSStringFromDataShape(_convDepthWise.outputShapeRef).UTF8String,
             NSStringFromDataShape(_convProject.outputShapeRef).UTF8String);
}

- (void)setLabel:(NSString *)label {
    [super setLabel:label];
    
    [_convExpand setLabel:[NSString stringWithFormat:@"%@_expand", label]];
    [_convDepthWise setLabel:[NSString stringWithFormat:@"%@_depthwise", label]];
    [_convProject setLabel:[NSString stringWithFormat:@"%@_project", label]];
    [_addition setLabel:[NSString stringWithFormat:@"%@_addition", label]];
}

- (void)setOffset:(MTLInt2)offset {
    _offset = offset;
    _convDepthWise.offset = offset;
}

#ifdef DEBUG
- (void)setVerbose:(int)verbose {
    [super setVerbose:verbose];
    [_convExpand setVerbose:verbose];
    [_convDepthWise setVerbose:verbose];
    [_convProject setVerbose:verbose];
}
#endif

- (void)setExpandDataSource:(MICNNKernelDataSource *)expandDataSource {
    _expandDataSource = expandDataSource;
    _convExpand.dataSource = expandDataSource;
}

- (void)setDepthWiseDataSource:(MICNNKernelDataSource *)depthWiseDataSource {
    _depthWiseDataSource = depthWiseDataSource;
    _convDepthWise.dataSource = depthWiseDataSource;
}

- (void)setProjectDataSource:(MICNNKernelDataSource *)projectDataSource {
    _projectDataSource = projectDataSource;
    _convProject.dataSource = projectDataSource;
}

- (MIConvolutionLayer *)expandComponent {
    return _convExpand;
}

- (MIConvolutionLayer *)depthWiseComponent {
    return _convDepthWise;
}

- (MIConvolutionLayer *)projectComponent {
    return _convProject;
}

- (void)removeAllTargets {
    [_lastNode removeAllTargets];
}

- (void)removeTarget:(id<MetalTensorInput>)targetToRemove {
    [_lastNode removeTarget:targetToRemove];
}

- (void)addTarget:(id<MetalTensorInput>)newTarget {
    [_lastNode addTarget:newTarget];
}

- (void)addTarget:(id<MetalTensorInput>)newTarget atTempImageIndex:(NSInteger)imageIndex {
    [_lastNode addTarget:newTarget atTempImageIndex:imageIndex];
}

- (void)setInputImage:(MITemporaryImage *)newInputImage atIndex:(NSInteger)imageIndex {
    [_convExpand setInputImage:newInputImage atIndex:0];
    [_addition setInputImage:newInputImage atIndex:0];
}

- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)cmdBuf {
    [_convExpand tempImageReadyAtIndex:0 commandBuffer:cmdBuf];
    [_addition tempImageReadyAtIndex:0 commandBuffer:cmdBuf];
}

#pragma mark - Management of the weights

- (BOOL)didLoadWeights {
    return [_convExpand didLoadWeights] && [_convDepthWise didLoadWeights] && [_convProject didLoadWeights];
}

- (void)loadWeights {
    [_convExpand loadWeights];
    [_convDepthWise loadWeights];
    [_convProject loadWeights];
}

- (void)loadWeightsList:(NSArray<NSString *> *)weightsList rangeList:(NSRange *)rangeList {
    [_convExpand loadWeights:weightsList[0] range:&rangeList[0]];
    [_convDepthWise loadWeights:weightsList[1] range:&rangeList[1]];
    [_convProject loadWeights:weightsList[2] range:&rangeList[2]];
}

MIInvertedResidualModule *MakeInvertedResidualModule(NSString *expand,
                                                     NSString *depthwise,
                                                     NSString *project,
                                                     KernelShape *kernels,
                                                     NeuronType *neurons,
                                                     DataShape *inputShape)
{
    assert(kernels != NULL);
    assert(neurons != NULL);
    assert(inputShape != NULL);
    
    MIInvertedResidualModule *module = [[MIInvertedResidualModule alloc] initWithInputShape:inputShape];
    npmemcpy(module.kernels, kernels, 3 * sizeof(KernelShape));
    npmemcpy(module.neurons, neurons, 3 * sizeof(NeuronType));

    [module setExpandDataSource:MakeDataSource(expand, &kernels[0], &neurons[0])];
    [module setDepthWiseDataSource:MakeDataSource(depthwise, &kernels[1], &neurons[1])];
    [module setProjectDataSource:MakeDataSource(project, &kernels[2], &neurons[2])];
    
    return module;
}

@end
