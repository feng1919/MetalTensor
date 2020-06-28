//
//  PortraitSegmentNet.h
//  MetalTensorDemo
//
//  Created by Feng Stone on 2019/9/25.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <MetalTensor/MetalNeuralNetwork.h>

NS_ASSUME_NONNULL_BEGIN

@protocol PortraitSegmentNetDelegate;
@interface PortraitSegmentNet : MetalNeuralNetwork

@property (nonatomic, weak) id<PortraitSegmentNetDelegate> delegate;

- (instancetype)init NS_DESIGNATED_INITIALIZER;
- (MTLUInt2)outputMaskSize;

@end

@protocol PortraitSegmentNetDelegate <NSObject>

@required
- (void)PortraitSegmentNet:(PortraitSegmentNet *)net predictResult:(float16_t *)result;

@end

extern NSNotificationName PortraitSegmentNetDidFinish;

NS_ASSUME_NONNULL_END
