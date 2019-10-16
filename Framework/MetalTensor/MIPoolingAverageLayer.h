//
//  MIPoolingAverageLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/24.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIPoolingAverageLayer : MetalTensorLayer

@property (nonatomic, assign) KernelShape kernel;

@property (nonatomic, assign) MPSOffset offset;

@end

NS_ASSUME_NONNULL_END
