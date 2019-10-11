//
//  MIL2NormalizationLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/9.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIL2NormalizationLayer : MetalTensorLayer

@property (nonatomic, assign) NSUInteger kernelWidth;
@property (nonatomic, assign) NSUInteger kernelHeight;
@property (nonatomic, assign) NSUInteger strideInPixelsX;
@property (nonatomic, assign) NSUInteger strideInPixelsY;
@property (nonatomic, assign) MPSOffset offset;

@end

NS_ASSUME_NONNULL_END
