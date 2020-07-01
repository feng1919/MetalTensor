//
//  RapidFaceDetectNet.h
//  RapidFaceDetectNet
//
//  Created by Feng Stone on 2019/11/13.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <MetalTensor/MetalTensor.h>

NS_ASSUME_NONNULL_BEGIN

@protocol RapidFaceDetectNetDelegate;
@interface RapidFaceDetectNet : MetalNeuralNetwork

@property (nonatomic, weak) id<RapidFaceDetectNetDelegate> delegate;
@property (nonatomic, strong, nullable) NSArray<SSDObject *> *objects;

- (instancetype)init NS_DESIGNATED_INITIALIZER;

@end

@protocol RapidFaceDetectNetDelegate <NSObject>

@required
- (void)RapidFaceDetectNet:(RapidFaceDetectNet *)net didFinishWithObjects:(NSArray<SSDObject *> * _Nullable)objects;

@end

NS_ASSUME_NONNULL_END
