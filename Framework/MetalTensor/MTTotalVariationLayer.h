//
//  MTTotalVariationLayer.h
//  MetalTensor
//
//  Created by Feng Stone on 2020/1/15.
//  Copyright © 2020 fengshi. All rights reserved.
//

/*
 *  Computing the total variation of image.
 *
 */
#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MTTotalVariationLayer : MetalTensorLayer

/*
 *  The scale of output value.
 *  Default by 1.0f.
 */
@property (nonatomic, assign) float alpha;

/*
 *  The scale of input values.
 *  Default by 255.0f;
 */
@property (nonatomic, assign) float scale;

@end

NS_ASSUME_NONNULL_END
