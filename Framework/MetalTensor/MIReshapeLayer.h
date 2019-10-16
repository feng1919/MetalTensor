//
//  MIReshapeLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/23.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIReshapeLayer : MetalTensorLayer

- (instancetype)initWithInputShape:(DataShape *)inputShape outputShape:(DataShape *)outputShape;

@end

MIReshapeLayer *MakeReshapeLayer(DataShape *inputShape, DataShape *outputShape);

NS_ASSUME_NONNULL_END
