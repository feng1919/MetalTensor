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

@interface MTTensorCache : NSObject

+ (MTTensorCache *)sharedCache;

- (NSInteger)registerReuseIdentifier;
- (void)unregisterReuseIdentifier:(NSInteger)identifier;
- (MetalTensor)fetchTensorWithShape:(DataShape *)shape source:(BackwardTarget _Nullable)source commandBuffer:(id<MTLCommandBuffer>)commandBuffer;

- (void)cacheTensor:(MetalTensor)tensor;

- (void)beginContextWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
- (void)endContextWithCommandBuffer:(id<MTLCommandBuffer>)commandBufferfer;

NSString *KeyForTensorType(DataShape *shape);

@end

NS_ASSUME_NONNULL_END
