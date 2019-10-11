//
//  MITransposeConvolutionLayer.h
//  MetalTensorDemo
//
//  Created by Feng Stone on 2019/9/24.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MITransposeConvolutionLayer : MetalTensorLayer <MetalTensorWeights>

@property (nonatomic, assign) MPSImageEdgeMode edgeMode;
@property (nonatomic, assign) MPSOffset offset;
@property (nonatomic, assign) MTLInt2 kernelOffset;
@property (nonatomic, strong) id<MPSCNNConvolutionDataSource> dataSource;

- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape
                  kernelDataSource:(id<MPSCNNConvolutionDataSource>)dataSource;

- (void)setOffsetWithX:(NSUInteger)x Y:(NSUInteger)y Z:(NSUInteger)z;

@end

NS_ASSUME_NONNULL_END
