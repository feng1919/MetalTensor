//
//  MIFullyConnectedLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"
#import "MIDataSource.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIFullyConnectedLayer : MetalTensorLayer <MetalTensorWeights>

@property (nonatomic, assign) KernelShape kernel;
@property (nonatomic, assign) NeuronType neuron;

@property (nonatomic, strong) MICNNKernelDataSource *dataSource;

@end

NS_ASSUME_NONNULL_END
