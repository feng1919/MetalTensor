//
//  MTChannelCompress.h
//  MetalTensor
//
//  Created by Feng Stone on 2020/1/21.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import "MPSImage+Extension.h"
#import "metal_tensor_structures.h"
#import "MTTensor.h"
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface MTChannelCompress : NSObject

@property (nonatomic, readonly) int numberOfChannels;
@property (nonatomic, assign) float alpha;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithNumberOfChannels:(int)numberOfChannels NS_DESIGNATED_INITIALIZER;
- (void)compile:(id<MTLDevice>)device;
- (void)compressOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer sourceTensor:(MetalTensor)src destinationTensor:(MetalTensor)dst;

@end

NS_ASSUME_NONNULL_END
