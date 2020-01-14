//
//  MTChannelReduce.h
//  MetalTensor
//
//  Created by Feng Stone on 2020/1/14.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import "MPSImage+Extension.h"
#import "metal_tensor_structures.h"
#import "MTTensor.h"
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface MTChannelReduce : NSObject

@property (nonatomic, readonly) ReduceType type;
@property (nonatomic, readonly) int numberOfChannels;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithReduceType:(ReduceType)type numberOfChannels:(int)numberOfChannels NS_DESIGNATED_INITIALIZER;
- (void)compile:(id<MTLDevice>)device;
- (void)reduceOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer sourceTensor:(MetalTensor)src destinationTensor:(MetalTensor)dst;

@end

NS_ASSUME_NONNULL_END
