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

- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape;

- (instancetype)initWithInputShape:(DataShape *)inputShape
                        interShape:(DataShape *)interShape
                       outputShape:(DataShape *)outputShape;

- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape
               depthwiseKernelData:(id<MPSCNNConvolutionDataSource>)dataSource1
                 projectKernelData:(id<MPSCNNConvolutionDataSource>)dataSource2;

- (instancetype)initWithInputShape:(DataShape *)inputShape
                        interShape:(DataShape *)interShape
                       outputShape:(DataShape *)outputShape
               depthwiseKernelData:(id<MPSCNNConvolutionDataSource>)dataSource1
                 projectKernelData:(id<MPSCNNConvolutionDataSource>)dataSource2;

- (MIConvolutionLayer *)depthwiseComponent;
- (MIConvolutionLayer *)projectComponent;

MISeparableConvolutionLayer *MakeSeparableConvolutionLayer(DataShape *input,
                                                           DataShape *output,
                                                           MICNNKernelDataSource *depthWiseData,
                                                           MICNNKernelDataSource *pointWiseData,
                                                           NSString * __nullable name);

@end

NS_ASSUME_NONNULL_END
