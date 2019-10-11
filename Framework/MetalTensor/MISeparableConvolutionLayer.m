//
//  MISeparableConvolutionLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/6.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MISeparableConvolutionLayer.h"

@interface MISeparableConvolutionLayer() {
    
    MIConvolutionLayer *_depthwise;
    MIConvolutionLayer *_project;
}

@end

@implementation MISeparableConvolutionLayer


- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape
{
    return [self initWithInputShape:inputShape interShape:inputShape outputShape:outputShape];
}

- (instancetype)initWithInputShape:(DataShape *)inputShape
                        interShape:(DataShape *)interShape
                       outputShape:(DataShape *)outputShape
{
    if (self = [super initWithInputShape:inputShape outputShape:outputShape]) {
        DB_TRACE(-_verbose+2, "\n%s init %s --> %s --> %s", self.labelUTF8,
                 NSStringFromDataShape(inputShape).UTF8String,
                 NSStringFromDataShape(interShape).UTF8String,
                 NSStringFromDataShape(outputShape).UTF8String);
        
        _depthwise = [[MIConvolutionLayer alloc] initWithInputShape:inputShape
                                                        outputShape:interShape];
        _project = [[MIConvolutionLayer alloc] initWithInputShape:interShape
                                                      outputShape:outputShape];
        [_depthwise addTarget:_project];
    }
    return self;
}

- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape
               depthwiseKernelData:(id<MPSCNNConvolutionDataSource>)dataSource1
                 projectKernelData:(id<MPSCNNConvolutionDataSource>)dataSource2
{
    return [self initWithInputShape:inputShape interShape:inputShape outputShape:outputShape depthwiseKernelData:dataSource1 projectKernelData:dataSource2];
}

- (instancetype)initWithInputShape:(DataShape *)inputShape
                        interShape:(DataShape *)interShape
                       outputShape:(DataShape *)outputShape
               depthwiseKernelData:(id<MPSCNNConvolutionDataSource>)dataSource1
                 projectKernelData:(id<MPSCNNConvolutionDataSource>)dataSource2
{
    if (self = [super initWithInputShape:inputShape outputShape:outputShape]) {
        DB_TRACE(-_verbose+2, "\n%s init %s --> %s --> %s", self.labelUTF8,
                 NSStringFromDataShape(inputShape).UTF8String,
                 NSStringFromDataShape(interShape).UTF8String,
                 NSStringFromDataShape(outputShape).UTF8String);
        
        _depthwise = [[MIConvolutionLayer alloc] initWithInputShape:inputShape
                                                       outputShape:interShape
                                                  kernelDataSource:dataSource1];
        _project = [[MIConvolutionLayer alloc] initWithInputShape:interShape
                                                     outputShape:outputShape
                                                kernelDataSource:dataSource2];
        [_depthwise addTarget:_project];
        
    }
    return self;
}

- (MIConvolutionLayer *)depthwiseComponent {
    return _depthwise;
}

- (MIConvolutionLayer *)projectComponent {
    return _project;
}

- (void)addTarget:(id<MetalTensorInput>)newTarget {
    [_project addTarget:newTarget];
}

- (void)addTarget:(id<MetalTensorInput>)newTarget atTempImageIndex:(NSInteger)imageIndex {
    [_project addTarget:newTarget atTempImageIndex:imageIndex];
}

- (void)removeTarget:(id<MetalTensorInput>)targetToRemove {
    [_project removeTarget:targetToRemove];
}

- (void)removeAllTargets {
    [_project removeAllTargets];
}

- (void)setLabel:(NSString *)label {
    [super setLabel:label];
    
    [_depthwise setLabel:[NSString stringWithFormat:@"%@_depthwise", label]];
    [_project setLabel:[NSString stringWithFormat:@"%@_project", label]];
}

#ifdef DEBUG
- (void)setVerbose:(int)verbose {
    [super setVerbose:verbose];
    
    [_depthwise setVerbose:verbose];
    [_project setVerbose:verbose];
}
#endif

- (void)setInputImage:(MITemporaryImage *)newInputImage atIndex:(NSInteger)imageIndex {
    [_depthwise setInputImage:newInputImage atIndex:imageIndex];
}

- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)cmdBuf {
    [_depthwise tempImageReadyAtIndex:imageIndex commandBuffer:cmdBuf];
}

#pragma mark - Management of the weights

- (BOOL)didLoadWeights {
    return [_depthwise didLoadWeights] && [_project didLoadWeights];
}

- (void)loadWeights {
    [_depthwise loadWeights];
    [_project loadWeights];
}

- (void)loadWeightsList:(NSArray<NSString *> *)weightsList rangeList:(NSRange *)rangeList
           kernelShapes:(KernelShape *)kernelShapes neuronTypes:(NeuronType *)neuronTypes depthWises:(BOOL *)depthWises {
    [_depthwise loadWeights:weightsList[0] range:&rangeList[0] kernelShape:&kernelShapes[0] neuronType:&neuronTypes[0] depthWise:depthWises?depthWises[0]:YES];
    [_project loadWeights:weightsList[1] range:&rangeList[1] kernelShape:&kernelShapes[1] neuronType:&neuronTypes[1] depthWise:depthWises?depthWises[1]:NO];
}

MISeparableConvolutionLayer *MakeSeparableConvolutionLayer(DataShape *input,
                                                           DataShape *output,
                                                           MICNNKernelDataSource *depthWiseData,
                                                           MICNNKernelDataSource *pointWiseData,
                                                           NSString *name)
{
    DataShape interShape = *input;
    if (depthWiseData.kernel.stride != 1) {
        interShape.row = ceilf((float)interShape.row/(float)depthWiseData.kernel.stride);
        interShape.column = ceilf((float)interShape.column/(float)depthWiseData.kernel.stride);
    }
    MISeparableConvolutionLayer *separableLayer = [[MISeparableConvolutionLayer alloc] initWithInputShape:input interShape:&interShape outputShape:output depthwiseKernelData:depthWiseData projectKernelData:pointWiseData];
    separableLayer.label = name;
    return separableLayer;
}

@end
