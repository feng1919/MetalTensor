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

- (NSInteger)registerReusePoolIdentifier {
    _reuseCounter ++;
    _reuseCacheMap[@(_reuseCounter)] = [[NSMutableDictionary alloc] init];
    return _reuseCounter;
}

- (void)unregisterReusePoolIdentifier:(NSInteger)identifier {
    [_reuseCacheMap removeObjectForKey:@(identifier)];
}


- (MetalTensor)fetchTensorWithShape1:(DataShape)shape commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    return [self fetchTensorWithShape1:shape dataType:MPSDataTypeFloat16 commandBuffer:commandBuffer];
}

- (MetalTensor)fetchTensorWithShape1:(DataShape)shape
                           dataType:(MPSDataType)dataType
                      commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    return [self fetchTensorWithShape:&shape dataType:dataType commandBuffer:commandBuffer];
}

- (MetalTensor)fetchTensorWithShape:(DataShape *)shape commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    return [self fetchTensorWithShape:shape dataType:MPSDataTypeFloat16 commandBuffer:commandBuffer];
}

- (MetalTensor)fetchTensorWithShape:(DataShape *)shape
                           dataType:(MPSDataType)dataType
                      commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    if (shape == NULL) {
        return nil;
    }
    
    NSAssert(DataShapeValid(shape), @"Invalid shape...");
    
    @synchronized (self) {
        MetalTensor tensor = nil;
        NSInteger reuseIdentifier = [commandBuffer.label integerValue];
        NSMutableDictionary *tensorCache = _reuseCacheMap[@(reuseIdentifier)];
        NSString *key = KeyForTensorType1(shape, dataType, 1);
        NSMutableSet<MetalTensor> *tensorSet = tensorCache[key];
        if (tensorSet.count > 0) {
            tensor = [tensorSet anyObject];
            [tensorSet removeObject:tensor];
        }
        else {
            tensor = [[MTTensor alloc] initWithShape:shape dataType:dataType numberOfImage:1];
            NSLog(@"Create a tensor: %@", KeyForTensorType1(shape, dataType, 1));
        }
        
        tensor.poolIdentifier = [commandBuffer.label integerValue];
        [tensor lock];
        [tensor newContentOnCommandBuffer:commandBuffer];
        return tensor;
    }
}

- (MetalMatrix)fetchMatrixWithRows:(int)rows
                           columns:(int)columns
                          dataType:(MPSDataType)dataType
                     commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
                         
    NSParameterAssert(rows > 0);
    NSParameterAssert(columns > 0);
     
    @synchronized (self) {
        MetalMatrix matrix = nil;
        NSInteger reuseIdentifier = [commandBuffer.label integerValue];
        NSMutableDictionary *cache = _reuseCacheMap[@(reuseIdentifier)];
        NSString *key = KeyForMatrixType(rows, columns, dataType);
        NSMutableSet<MetalMatrix> *matrixSet = cache[key];
        if (matrixSet.count > 0) {
            matrix = [matrixSet anyObject];
            [matrixSet removeObject:matrix];
        }
        else {
            matrix = [[MTMatrix alloc] initWithRows:rows columns:columns dataType:dataType];
            NSLog(@"Create a matrix: %@", KeyForMatrixType(rows, columns, dataType));
        }

        matrix.poolIdentifier = [commandBuffer.label integerValue];
        [matrix lock];
        [matrix newContentOnCommandBuffer:commandBuffer];
        return matrix;
    }
}

- (void)cacheResource:(id<MTResource>)resource {
    NSParameterAssert(resource);
    
    if (![_reuseCacheMap.allKeys containsObject:@(resource.poolIdentifier)]) {
        return;
    }
    
    @synchronized (self) {
        NSMutableDictionary *pool = _reuseCacheMap[@(resource.poolIdentifier)];
        NSString *key = resource.reuseIdentifier;
        NSMutableSet<id<MTResource>> *set = pool[key];
        if (set == nil) {
            set = [[NSMutableSet alloc] init];
            pool[key] = set;
        }
        [set addObject:resource];
    }
}

- (void)beginContextWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    NSInteger poolIdentifier = [commandBuffer.label integerValue];
    NSMutableDictionary *tensorCache = _reuseCacheMap[@(poolIdentifier)];
    NSMutableArray *tensorList = [NSMutableArray array];
    NSMutableArray *matrixList = [NSMutableArray array];
    for (NSSet *set in tensorCache.allValues) {
        id obj = set.anyObject;
        if ([obj isKindOfClass:[MTTensor class]]) {
            for (MetalTensor tensor in set.allObjects) {
                if (tensor.poolIdentifier == poolIdentifier) {
                    [tensorList addObject:tensor.imageDescriptor];
                }
            }
        }
        else if ([obj isKindOfClass:[MTMatrix class]]){
            for (MetalMatrix matrix in set.allObjects) {
                if (matrix.poolIdentifier == poolIdentifier) {
                    [matrixList addObject:matrix.matrixDescriptor];
                }
            }
        }
        else {
            NSAssert(NO, @"Unsupported resource type: %@", obj);
        }
    }
    
    if (tensorList.count > 0) {
        [MPSTemporaryImage prefetchStorageWithCommandBuffer:commandBuffer imageDescriptorList:tensorList];
    }
    if (matrixList.count > 0) {
        [MPSTemporaryMatrix prefetchStorageWithCommandBuffer:commandBuffer matrixDescriptorList:matrixList];
    }
}

- (void)endContextWithCommandBuffer:(id<MTLCommandBuffer>)commandBufferfer {
    NSInteger identifier = [commandBufferfer.label integerValue];
    NSMutableDictionary *tensorCache = _reuseCacheMap[@(identifier)];
    for (NSSet *set in tensorCache.allValues) {
        for (id<MTResource> resource in set.allObjects) {
            if (resource.poolIdentifier == identifier) {
                [resource deleteContent];
            }
        }
    }
}

@end
