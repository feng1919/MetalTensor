//
//  MIBatchNormalizationLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIBatchNormalizationLayer : MetalTensorLayer

@property (nonatomic, assign) MPSImageEdgeMode edgeMode;
@property (nonatomic, strong) id<MPSCNNBatchNormalizationDataSource> dataSource;
@property (nonatomic, assign) float epsilon;//default 0.001

- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape
                  kernelDataSource:(id<MPSCNNBatchNormalizationDataSource>)dataSource;

@end

NS_ASSUME_NONNULL_END
