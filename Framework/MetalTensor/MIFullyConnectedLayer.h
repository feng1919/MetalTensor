//
//  MIFullyConnectedLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIFullyConnectedLayer : MetalTensorLayer <MetalTensorWeights>

@property (nonatomic, strong) id<MPSCNNConvolutionDataSource> dataSource;

- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape
                  kernelDataSource:(id<MPSCNNConvolutionDataSource>)dataSource;

@end

NS_ASSUME_NONNULL_END
