//
//  MTTensorCache.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MTTensorCache.h"
#import "MPSImage+Extension.h"

static MTTensorCache *_sharedTensorCache = nil;
static int _reuseCounter = 0;

@interface MTTensorCache() {
    NSMutableDictionary<NSNumber *, NSMutableDictionary *> *_reuseCacheMap;
}

@end

@implementation MTTensorCache

+ (void)initialize {
    if (self == [MTTensorCache class]) {
        _sharedTensorCache = [[MTTensorCache alloc] init];
    }
}

+ (MTTensorCache *)sharedCache {
    return _sharedTensorCache;
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

- (MetalTensor)fetchTensorWithShape:(DataShape *)shape source:(BackwardTarget)source commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    if (shape == NULL) {
        return nil;
    }
    
    NSAssert(DataShapeValid(shape), @"Invalid shape...");
    
    @synchronized (self) {
        MetalTensor tensor = nil;
        NSInteger reuseIdentifier = [commandBuffer.label integerValue];
        NSMutableDictionary *tensorCache = _reuseCacheMap[@(reuseIdentifier)];
        NSString *key = KeyForTensorType(shape);
        NSMutableSet<MetalTensor> *tensorSet = tensorCache[key];
        if (tensorSet.count > 0) {
            tensor = [tensorSet anyObject];
            [tensorSet removeObject:tensor];
        }
        else {
            tensor = [[MTTensor alloc] initWithShape:shape];
            NSLog(@"Create a tensor: %@", KeyForTensorType(shape));
        }
        
        tensor.reuseIdentifier = [commandBuffer.label integerValue];
        tensor.source = source;
        [tensor lock];
        return tensor;
    }
}

- (void)cacheTensor:(MetalTensor)tensor {
    if (tensor == nil) {
        return;
    }
    
    if (![_reuseCacheMap.allKeys containsObject:@(tensor.reuseIdentifier)]) {
        return;
    }
    
    @synchronized (self) {
        NSMutableDictionary *tensorCache = _reuseCacheMap[@(tensor.reuseIdentifier)];
        NSString *key = KeyForTensorType([tensor shape]);
        NSMutableSet<MetalTensor> *tensorSet = tensorCache[key];
        if (tensorSet == nil) {
            tensorSet = [[NSMutableSet alloc] init];
            tensorCache[key] = tensorSet;
        }
        [tensorSet addObject:tensor];
        tensor.source = nil;
    }
}

- (void)beginContextWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    NSInteger identifier = [commandBuffer.label integerValue];
    NSMutableDictionary *tensorCache = _reuseCacheMap[@(identifier)];
    NSMutableArray *tensorList = [NSMutableArray array];
    for (NSSet *set in tensorCache.allValues) {
        for (MetalTensor tensor in set.allObjects) {
            if (tensor.reuseIdentifier == identifier) {
                [tensorList addObject:tensor.imageDescriptor];
            }
        }
    }
    [MPSTemporaryImage prefetchStorageWithCommandBuffer:commandBuffer imageDescriptorList:tensorList];
}

- (void)endContextWithCommandBuffer:(id<MTLCommandBuffer>)commandBufferfer {
    NSInteger identifier = [commandBufferfer.label integerValue];
    NSMutableDictionary *tensorCache = _reuseCacheMap[@(identifier)];
    for (NSSet *set in tensorCache.allValues) {
        for (MetalTensor tensor in set.allObjects) {
            if (tensor.reuseIdentifier == identifier) {
                [tensor deleteContent];
            }
        }
    }
}

NSString *KeyForTensorType(DataShape *shape) {
    return [NSString stringWithFormat:@"[ROW %d][COLUMN %d][DEPTH %d]", shape->row, shape->column, shape->depth];
}

@end
