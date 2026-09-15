//
//  PortraitSegmentFilter.h
//  MetalTensorDemo
//
//  Created by Feng Stone on 2019/9/27.
//  Copyright © 2019 fengshi. All rights reserved.
//

#if __has_include(<MetalImage/MetalImage.h>)
#import <MetalImage/MetalImage.h>
#else
#import "MetalImage.h"
#endif
#import "PortraitSegmentNet.h"

NS_ASSUME_NONNULL_BEGIN

/*
 *  1. Accept a MTLTexture from MetalImage pipeline, then resize it to
 *  the neural network input size and feed into the network.
 *  2. Obtain the segmenting mask, as the network processing result,
 *  and render the segmentation result.
 *
 */

@interface PortraitSegmentFilter : MetalImageFilter <PortraitSegmentNetDelegate>

- (void)createNet;

@end

NS_ASSUME_NONNULL_END
