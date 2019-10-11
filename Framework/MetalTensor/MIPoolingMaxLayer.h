//
//  MIPoolingMaxLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIPoolingMaxLayer : MetalTensorLayer

@property (nonatomic, assign) NSUInteger kernelWidth;
@property (nonatomic, assign) NSUInteger kernelHeight;
@property (nonatomic, assign) NSUInteger strideInPixelsX;
@property (nonatomic, assign) NSUInteger strideInPixelsY;
@property (nonatomic, assign) MPSOffset offset;

- (void)setOffsetWithX:(NSInteger)x Y:(NSInteger)y Z:(NSInteger)z;

@end

NS_ASSUME_NONNULL_END
