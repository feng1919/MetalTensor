//
//  MIDropoutLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/24.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIDropoutLayer : MetalTensorLayer

@property (nonatomic, assign) float keepProbability; //default 0.999

@end

NS_ASSUME_NONNULL_END
