//
//  MITemporaryImageCache.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MITemporaryImageCache.h"
#import "MPSImage+Extension.h"

static MITemporaryImageCache *_sharedTemporaryImageCache = nil;
static int _reuseCounter = 0;

@interface MITemporaryImageCache() {
    NSMutableDictionary<NSNumber *, NSMutableDictionary *> *_reuseCacheMap;
}

@end

@implementation MITemporaryImageCache

+ (void)initialize {
    if (self == [MITemporaryImageCache class]) {
        _sharedTemporaryImageCache = [[MITemporaryImageCache alloc] init];
    }
}

+ (MITemporaryImageCache *)sharedCache {
    return _sharedTemporaryImageCache;
}

- (instancetype)init {
    if (self = [super init]) {
        _reuseCacheMap = [[NSMutableDictionary alloc] init];
    }
    return self;
}

- (NSInteger)registerReuseIdentifier {
    _reuseCounter ++;
    _reuseCacheMap[@(_reuseCounter)] = [[NSMutableDictionary alloc] init];
    return _reuseCounter;
}

- (void)unregisterReuseIdentifier:(NSInteger)identifier {
    [_reuseCacheMap removeObjectForKey:@(identifier)];
}

- (MITemporaryImage *)fetchTemporaryImageWithShape:(DataShape *)imageParameters commandBuffer:(nonnull id<MTLCommandBuffer>)commandBuffer {
    if (imageParameters == NULL) {
        return nil;
    }
    
    NSAssert(DataShapeValid(imageParameters), @"Invalid image type...");
    
    @synchronized (self) {
        MITemporaryImage *image = nil;
        NSInteger reuseIdentifier = [commandBuffer.label integerValue];
        NSMutableDictionary *tempImageCache = _reuseCacheMap[@(reuseIdentifier)];
        NSString *key = KeyForImageType(imageParameters);
        NSMutableSet<MITemporaryImage *> *imageSet = tempImageCache[key];
        if (imageSet.count > 0) {
            image = [imageSet anyObject];
            [imageSet removeObject:image];
        }
        else {
            image = [[MITemporaryImage alloc] initWithShape:imageParameters];
            NSLog(@"Create a temporary image: %@", KeyForImageType(imageParameters));
        }
        
        image.reuseIdentifier = [commandBuffer.label integerValue];
        [image lock];
        return image;
    }
}

- (void)cacheImage:(MITemporaryImage *)image {
    if (image == nil) {
        return;
    }
    
    if (![_reuseCacheMap.allKeys containsObject:@(image.reuseIdentifier)]) {
        return;
    }
    
    @synchronized (self) {
        NSMutableDictionary *tempImageCache = _reuseCacheMap[@(image.reuseIdentifier)];
        NSString *key = KeyForImageType([image shape]);
        NSMutableSet<MITemporaryImage *> *imageSet = tempImageCache[key];
        if (imageSet == nil) {
            imageSet = [[NSMutableSet alloc] init];
            tempImageCache[key] = imageSet;
        }
        [imageSet addObject:image];
    }
}

- (void)beginContextWithCommandBuffer:(id<MTLCommandBuffer>)cmdBuf {
    NSInteger identifier = [cmdBuf.label integerValue];
    NSMutableDictionary *tempImageCache = _reuseCacheMap[@(identifier)];
    NSMutableArray *imageList = [NSMutableArray array];
    for (NSSet *set in tempImageCache.allValues) {
        for (MITemporaryImage *image in set.allObjects) {
            if (image.reuseIdentifier == identifier) {
                [imageList addObject:image.imageDescriptor];
            }
        }
    }
    [MPSTemporaryImage prefetchStorageWithCommandBuffer:cmdBuf imageDescriptorList:imageList];
}

- (void)endContextWithCommandBuffer:(id<MTLCommandBuffer>)cmdBuffer {
    NSInteger identifier = [cmdBuffer.label integerValue];
    NSMutableDictionary *tempImageCache = _reuseCacheMap[@(identifier)];
    for (NSSet *set in tempImageCache.allValues) {
        for (MITemporaryImage *image in set.allObjects) {
            if (image.reuseIdentifier == identifier) {
                [image deleteTemporaryImage];
            }
        }
    }
}

NSString *KeyForImageType(DataShape *type) {
    return [NSString stringWithFormat:@"[ROW %d][COLUMN %d][DEPTH %d]", type->row, type->column, type->depth];
}

#pragma mark - Concate images

//+ (MITemporaryImage *)temporaryImageForConcatenation:(NSArray<MITemporaryImage *> *)images {
//    DataShape **shapeList = malloc(images.count * sizeof(DataShape *));
//    for (int i = 0; i < images.count; i++) {
//        shapeList[i] = [images[i] shape];
//    }
//    DataShape dstShape = ConcatenateShapes1(shapeList, (int)images.count, NULL, true);
//    free(shapeList);
//
//    return [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&dstShape];
//}

@end
