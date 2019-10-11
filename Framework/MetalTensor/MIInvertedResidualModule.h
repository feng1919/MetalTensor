//
//  MIInvertedResidualModule.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/21.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"
#import "MIConvolutionLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIInvertedResidualModule : MetalTensorLayer <MetalTensorWeights>

@property (nonatomic, strong) id<MPSCNNConvolutionDataSource> expandDataSource;
@property (nonatomic, strong) id<MPSCNNConvolutionDataSource> depthWiseDataSource;
@property (nonatomic, strong) id<MPSCNNConvolutionDataSource> projectDataSource;

- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape
                      dwInputShape:(DataShape *)dwInputShape
                     dwOutputShape:(DataShape *)dwOutputShape;

- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape
                      dwInputShape:(DataShape *)dwInputShape
                     dwOutputShape:(DataShape *)dwOutputShape
                  expandDataSource:(id<MPSCNNConvolutionDataSource>)expandDataSource
               depthWiseDataSource:(id<MPSCNNConvolutionDataSource>)depthWiseDataSource
                 projectDataSource:(id<MPSCNNConvolutionDataSource>)projectDataSource;

- (MIConvolutionLayer *)expandComponent;
- (MIConvolutionLayer *)depthWiseComponent;
- (MIConvolutionLayer *)projectComponent;

MIInvertedResidualModule *MakeInvertedResidualModule(NSString *expand,
                                                     NSString *depthwise,
                                                     NSString *project,
                                                     KernelShape * _Nonnull * _Nonnull kernel,
                                                     NeuronType * _Nonnull * _Nonnull neuron,
                                                     DataShape *_Nonnull * _Nonnull shape);

@end

NS_ASSUME_NONNULL_END
