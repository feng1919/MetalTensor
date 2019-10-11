//
//  MITemporaryImageCache.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MITemporaryImage.h"

NS_ASSUME_NONNULL_BEGIN

@interface MITemporaryImageCache : NSObject

+ (MITemporaryImageCache *)sharedCache;

- (NSInteger)registerReuseIdentifier;
- (void)unregisterReuseIdentifier:(NSInteger)identifier;
- (MITemporaryImage *)fetchTemporaryImageWithShape:(DataShape *)imageParameters commandBuffer:(id<MTLCommandBuffer>)commandBuffer;

- (void)cacheImage:(MITemporaryImage *)image;

- (void)beginContextWithCommandBuffer:(id<MTLCommandBuffer>)cmdBuf;
- (void)endContextWithCommandBuffer:(id<MTLCommandBuffer>)cmdBuffer;

//+ (MITemporaryImage *)temporaryImageForConcatenation:(NSArray<MITemporaryImage *> *)images;

NSString *KeyForImageType(DataShape *type);

@end

NS_ASSUME_NONNULL_END
