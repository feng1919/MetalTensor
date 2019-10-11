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

@interface MIInvertedResidualModule() {
    
    MIConvolutionLayer *_convExpand;
    MIConvolutionLayer *_convDepthWise;
    MIConvolutionLayer *_convProject;
    MIArithmeticLayer *_addition;
}

@end

@implementation MIInvertedResidualModule

- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape
                      dwInputShape:(DataShape *)dwInputShape
                     dwOutputShape:(DataShape *)dwOutputShape {
    if (self = [super initWithInputShape:inputShape outputShape:outputShape]) {
        
        _convExpand = [[MIConvolutionLayer alloc] initWithInputShape:inputShape
                                                        outputShape:dwInputShape];
        _convDepthWise = [[MIConvolutionLayer alloc] initWithInputShape:dwInputShape
                                                           outputShape:dwOutputShape];
        _convProject = [[MIConvolutionLayer alloc] initWithInputShape:dwOutputShape
                                                         outputShape:outputShape];
        [_convExpand addTarget:_convDepthWise];
        [_convDepthWise addTarget:_convProject];
        
        if (DataShapesTheSame(inputShape, outputShape)) {
            _addition = [MIArithmeticLayer additionArithmeticLayerWithDataShape:outputShape];
            [_convProject addTarget:_addition atTempImageIndex:1];
        }
        
        DB_TRACE(-_verbose+2, "\n%s init %s --> %s --> %s --> %s", self.labelUTF8, NSStringFromDataShape(inputShape).UTF8String,
                 NSStringFromDataShape(outputShape).UTF8String, NSStringFromDataShape(dwInputShape).UTF8String, NSStringFromDataShape(dwOutputShape).UTF8String);
    }
    return self;
}

- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape
                      dwInputShape:(DataShape *)dwInputShape
                     dwOutputShape:(DataShape *)dwOutputShape
                  expandDataSource:(id<MPSCNNConvolutionDataSource>)expandDataSource
               depthWiseDataSource:(id<MPSCNNConvolutionDataSource>)depthWiseDataSource
                 projectDataSource:(id<MPSCNNConvolutionDataSource>)projectDataSource {
    if (self = [super initWithInputShape:inputShape outputShape:outputShape]) {
        
        _convExpand = [[MIConvolutionLayer alloc] initWithInputShape:inputShape
                                                        outputShape:dwInputShape
                                                   kernelDataSource:expandDataSource];
        _convDepthWise = [[MIConvolutionLayer alloc] initWithInputShape:dwInputShape
                                                           outputShape:dwOutputShape
                                                      kernelDataSource:depthWiseDataSource];
        _convProject = [[MIConvolutionLayer alloc] initWithInputShape:dwOutputShape
                                                         outputShape:outputShape
                                                    kernelDataSource:projectDataSource];
        [_convExpand addTarget:_convDepthWise];
        [_convDepthWise addTarget:_convProject];
        
        if (DataShapesTheSame(inputShape, outputShape)) {
            _addition = [MIArithmeticLayer additionArithmeticLayerWithDataShape:outputShape];
            [_convProject addTarget:_addition atTempImageIndex:1];
        }
        
        DB_TRACE(-_verbose+2, "\n%s init %s --> %s --> %s --> %s", self.labelUTF8, NSStringFromDataShape(inputShape).UTF8String,
                 NSStringFromDataShape(outputShape).UTF8String, NSStringFromDataShape(dwInputShape).UTF8String, NSStringFromDataShape(dwOutputShape).UTF8String);
        
        self.expandDataSource = expandDataSource;
        self.depthWiseDataSource = depthWiseDataSource;
        self.projectDataSource = projectDataSource;
    }
    return self;
}

- (void)setLabel:(NSString *)label {
    [super setLabel:label];
    
    [_convExpand setLabel:[NSString stringWithFormat:@"%@_expand", label]];
    [_convDepthWise setLabel:[NSString stringWithFormat:@"%@_depthwise", label]];
    [_convProject setLabel:[NSString stringWithFormat:@"%@_project", label]];
    [_addition setLabel:[NSString stringWithFormat:@"%@_addition", label]];
}

#ifdef DEBUG
- (void)setVerbose:(int)verbose {
    [super setVerbose:verbose];
    [_convExpand setVerbose:verbose];
    [_convDepthWise setVerbose:verbose];
    [_convProject setVerbose:verbose];
}
#endif

- (void)setExpandDataSource:(id<MPSCNNConvolutionDataSource>)expandDataSource {
    _expandDataSource = expandDataSource;
    _convExpand.dataSource = expandDataSource;
}

- (void)setDepthWiseDataSource:(id<MPSCNNConvolutionDataSource>)depthWiseDataSource {
    _depthWiseDataSource = depthWiseDataSource;
    _convDepthWise.dataSource = depthWiseDataSource;
}

- (void)setProjectDataSource:(id<MPSCNNConvolutionDataSource>)projectDataSource {
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
    if (_addition) {
        [_addition removeAllTargets];
    }
    else {
        [_convProject removeAllTargets];
    }
}

- (void)removeTarget:(id<MetalTensorInput>)targetToRemove {
    if (_addition) {
        [_addition removeTarget:targetToRemove];
    }
    else {
        [_convProject removeTarget:targetToRemove];
    }
}

- (void)addTarget:(id<MetalTensorInput>)newTarget {
    if (_addition) {
        [_addition addTarget:newTarget];
    }
    else {
        [_convProject addTarget:newTarget];
    }
}

- (void)addTarget:(id<MetalTensorInput>)newTarget atTempImageIndex:(NSInteger)imageIndex {
    if (_addition) {
        [_addition addTarget:newTarget atTempImageIndex:imageIndex];
    }
    else {
        [_convProject addTarget:newTarget atTempImageIndex:imageIndex];
    }
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

- (void)loadWeightsList:(NSArray<NSString *> *)weightsList rangeList:(NSRange *)rangeList
           kernelShapes:(KernelShape *)kernelShapes neuronTypes:(NeuronType *)neuronTypes depthWises:(BOOL *)depthWises {
    [_convExpand loadWeights:weightsList[0] range:&rangeList[0] kernelShape:&kernelShapes[0] neuronType:&neuronTypes[0] depthWise:depthWises?depthWises[0]:NO];
    [_convDepthWise loadWeights:weightsList[1] range:&rangeList[1] kernelShape:&kernelShapes[1] neuronType:&neuronTypes[1] depthWise:depthWises?depthWises[1]:YES];
    [_convProject loadWeights:weightsList[2] range:&rangeList[2] kernelShape:&kernelShapes[2] neuronType:&neuronTypes[2] depthWise:depthWises?depthWises[2]:NO];
}

MIInvertedResidualModule *MakeInvertedResidualModule(NSString *expand,
                                                     NSString *depthwise,
                                                     NSString *project,
                                                     KernelShape **kernel,
                                                     NeuronType **neuron,
                                                     DataShape **shape)
{
    MICNNKernelDataSource *data_expand = MakeDataSource(expand, kernel[0], neuron[0]);
    MICNNKernelDataSource *data_depthwise = MakeDataSource(depthwise, kernel[1], neuron[1]);
    MICNNKernelDataSource *data_project = MakeDataSource(project, kernel[2], neuron[2]);
    
    MIInvertedResidualModule *module = [[MIInvertedResidualModule alloc] initWithInputShape:shape[0] outputShape:shape[3] dwInputShape:shape[1] dwOutputShape:shape[2] expandDataSource:data_expand depthWiseDataSource:data_depthwise projectDataSource:data_project];
    return module;
}

@end
