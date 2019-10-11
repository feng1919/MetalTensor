//
//  MetalTensorNeuronLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/5.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MetalTensorNeuronLayer : MetalTensorLayer

@property (nonatomic, readonly) NeuronType neuronType;

- (instancetype)initWithDataShape:(DataShape *)dataShape neuronType:(NeuronType)neuronType;

@end

NS_ASSUME_NONNULL_END
