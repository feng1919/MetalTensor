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

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithInputShape:(DataShape *)inputShape NS_UNAVAILABLE;
- (instancetype)initWithInputShapes:(DataShape *)inputShapes size:(int)size NS_UNAVAILABLE;
- (instancetype)initWithInputShapes1:(DataShape *_Nonnull*_Nonnull)inputShapes size:(int)size NS_UNAVAILABLE;
- (instancetype)initWithInputShape:(DataShape *)inputShape outputShape:(DataShape *)outputShape NS_DESIGNATED_INITIALIZER;

@end

MIReshapeLayer *MakeReshapeLayer(DataShape *inputShape, DataShape *outputShape);

NS_ASSUME_NONNULL_END
