//
//  MTTensorCache.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MetalTensorProtocols.h"
#import "MTTensor.h"

NS_ASSUME_NONNULL_BEGIN

/*
 *  MTTensor is a container for data on GPU, which content is expensive
 *  to create, and MTTensorCache holds the data when it is out of using
 *  and if asked for a tensor with the same shape, then it will be reused.
 *  NOTE: The content of MTTensor is temporary and has the same life cycle
 *  to MTLCommandBuffer object.
 *
 */

@interface MTTensorCache : NSObject

+ (MTTensorCache *)sharedCache;

- (NSInteger)registerReuseIdentifier;
- (void)unregisterReuseIdentifier:(NSInteger)identifier;

- (MetalTensor)fetchTensorWithShape:(DataShape *)shape
                      commandBuffer:(id<MTLCommandBuffer>)commandBuffer;

- (MetalTensor)fetchTensorWithShape1:(DataShape)shape
                       commandBuffer:(id<MTLCommandBuffer>)commandBuffer;

- (MetalTensor)fetchTensorWithShape:(DataShape *)shape
                         dataFormat:(TensorDataFormat)dataFormat
                        numberOfImages:(NSUInteger)numberOfImages
                      commandBuffer:(id<MTLCommandBuffer>)commandBuffer;

- (void)cacheTensor:(MetalTensor)tensor;

- (void)beginContextWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
- (void)endContextWithCommandBuffer:(id<MTLCommandBuffer>)commandBufferfer;

NSString *KeyForTensorType(DataShape *shape, TensorDataFormat dataFormat);
NSString *KeyForTensorType1(DataShape *shape, TensorDataFormat dataFormat, NSUInteger numberOfImages);

@end

NS_ASSUME_NONNULL_END
