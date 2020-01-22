//
//  MTMeanSquaredErrorLayer.h
//  MetalTensor
//
//  Created by Feng Stone on 2020/1/14.
//  Copyright © 2020 fengshi. All rights reserved.
//

/*
 *  Compute the mean squared error of two tensors.
 *
 */

#import "MetalTensorLayer.h"
#import "MTImageTensor.h"

NS_ASSUME_NONNULL_BEGIN

@interface MTMeanSquaredErrorLayer : MetalTensorLayer

@property (nonatomic, assign) float alpha;
@property (nonatomic, strong, nullable) MTImageTensor *secondaryImage;

@end

NS_ASSUME_NONNULL_END
