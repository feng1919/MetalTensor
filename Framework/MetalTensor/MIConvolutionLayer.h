//
//  MIConvolutionLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIConvolutionLayer : MetalTensorLayer <MetalTensorWeights>

@property (nonatomic, assign) MPSImageEdgeMode edgeMode;
@property (nonatomic, assign) MPSOffset offset;
@property (nonatomic, strong) id<MPSCNNConvolutionDataSource> dataSource;

- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape
                  kernelDataSource:(id<MPSCNNConvolutionDataSource>)dataSource;

- (void)setOffsetWithX:(NSUInteger)x Y:(NSUInteger)y Z:(NSUInteger)z;

MIConvolutionLayer *MakeConvolutionLayer(NSString *module_name,
                                         KernelShape *k,
                                         NeuronType *n,
                                         DataShape *input,
                                         DataShape *output);

@end

NS_ASSUME_NONNULL_END
