//
//  MetalTensorInput.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MTTensor.h"

@protocol MTForwardDelegate <NSObject>

@required
//  The data shape of tensor forward propgation.
- (DataShape *)dataShapeRef;

- (void)imageReadyAtIndex:(NSInteger)imageIndex onCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
- (void)setImage:(MetalTensor)newImage atIndex:(NSInteger)imageIndex;
- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;

@optional
- (void)receiveImageAtIndex:(NSInteger)index;
- (BOOL)isAllImagesReceived;
- (void)resetImagesReceivedFlags;

- (NSInteger)nextAvailableImageIndex;
- (void)reserveImageIndex:(NSInteger)index;
- (void)releaseImageIndex:(NSInteger)index;

@end

typedef id<MTForwardDelegate> ForwardTarget;




@protocol MTBackwardDelegate <NSObject>

@required
- (void)setGradient:(MetalTensor)newGradient forwardTarget:(ForwardTarget)target;
- (void)gradientReadyFromForwardTarget:(ForwardTarget)target onCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;

@optional
- (BOOL)isAllGradientsReceived;

@end

typedef id<MTBackwardDelegate> BackwardTarget;




@protocol MetalTensorWeights <NSObject>

@required
- (void)loadWeights;
- (BOOL)didLoadWeights;

@optional

- (void)loadWeights:(NSData *)weights;

- (void)loadWeights:(NSString *)weights
              range:(NSRange *)range;

- (void)loadWeightsList:(NSArray<NSString *> *)weightsList
              rangeList:(NSRange *)rangeList;

@end
