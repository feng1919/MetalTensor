//
//  MISeparableConvolutionLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/6.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"
#import "MIConvolutionLayer.h"
#import "MIDataSource.h"

NS_ASSUME_NONNULL_BEGIN

@interface MISeparableConvolutionLayer : MetalTensorLayer <MetalTensorWeights>

@property (nonatomic, strong) MICNNKernelDataSource *dataSourceDepthWise;
@property (nonatomic, strong) MICNNKernelDataSource *dataSourceProject;

@property (nonatomic, readonly) KernelShape *kernels;
@property (nonatomic, readonly) NeuronType *neurons;
@property (nonatomic, assign) MTPaddingMode padding;
@property (nonatomic, assign) MTLInt2 offset;

- (MIConvolutionLayer *)depthwiseComponent;
- (MIConvolutionLayer *)projectComponent;

MISeparableConvolutionLayer *MakeSeparableConvolutionLayer(DataShape *input,
                                                           KernelShape *kernels,
                                                           NeuronType *neurons,
                                                           MICNNKernelDataSource *depthWiseData,
                                                           MICNNKernelDataSource *pointWiseData,
                                                           NSString * __nullable name);

@end

NS_ASSUME_NONNULL_END
