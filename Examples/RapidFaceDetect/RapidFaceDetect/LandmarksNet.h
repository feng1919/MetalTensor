//
//  LandmarksNet.h
//  RapidFaceDetect
//
//  Created by Feng Stone on 2020/7/25.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import <MetalTensor/MetalTensor.h>

NS_ASSUME_NONNULL_BEGIN

@protocol LandmarksNetDelegate;
@interface LandmarksNet : MetalNeuralNetwork

@property (nonatomic, weak) id<LandmarksNetDelegate> delegate;

- (instancetype)init NS_DESIGNATED_INITIALIZER;

@end

@protocol LandmarksNetDelegate <NSObject>

@required
- (void)LandmarksNet:(LandmarksNet *)net didFinishWithPoints:(float32_t *)points;

@end

NS_ASSUME_NONNULL_END
