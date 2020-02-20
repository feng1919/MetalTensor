//
//  MetalTensorNode.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorNode.h"

@implementation MetalTensorNode

//#ifdef DEBUG
static int _METAL_VERBOSE = 0;
+ (void)setVerbose:(int)verbose {
    _METAL_VERBOSE = verbose;
}

- (const char *)labelUTF8 {
    return self.label.UTF8String;
}
//#endif

- (instancetype)init {
    if (!(self = [super init]))
    {
        return nil;
    }
    
//#ifdef DEBUG
    _verbose = _METAL_VERBOSE;
//#endif

    _needBackward = NO;
    _targets = [[NSMutableArray alloc] init];
    _targetIndices = [[NSMutableArray alloc] init];
    
    return self;
}

- (void)dealloc {
    [self removeAllTargets];
}

- (NSString *)label {
    return _label?:NSStringFromClass([self class]);
}

- (NSString *)description {
    return self.label;
}

- (void)compile:(id<MTLDevice>)device {
    // Preparing for the computation.
    _device = device;
}

#pragma mark - FORWARD

- (NSArray<ForwardTarget> *)targets {
    @synchronized (self) {
        return [NSArray arrayWithArray:_targets];
    }
}

- (void)addTarget:(ForwardTarget)newTarget {
    @synchronized (self) {
        NSInteger nextAvailableImageIndex = 0;
        if ([newTarget respondsToSelector:@selector(nextAvailableImageIndex)]) {
            nextAvailableImageIndex = [newTarget nextAvailableImageIndex];
        }
        
        [self addTarget:newTarget atIndex:nextAvailableImageIndex];
    }
}

- (void)addTarget:(ForwardTarget)newTarget atIndex:(NSInteger)imageIndex {
    @synchronized (self) {
        if ([_targets containsObject:newTarget]) {
            return;
        }
        
        [_targets addObject:newTarget];
        [_targetIndices addObject:@(imageIndex)];
        
        if ([newTarget respondsToSelector:@selector(reserveImageIndex:)]) {
            [newTarget reserveImageIndex:imageIndex];
        }
        
        DB_TRACE(-_verbose+2, "\n%s add %s at %ld", self.labelUTF8, [newTarget description].UTF8String, imageIndex);
    }
}

- (void)removeTarget:(ForwardTarget)targetToRemove {
    @synchronized (self) {
        if(![_targets containsObject:targetToRemove]) {
            return;
        }
        
        NSInteger indexOfObject = [_targets indexOfObject:targetToRemove];
        NSInteger imageIndex = [[_targetIndices objectAtIndex:indexOfObject] integerValue];
        
        [_targetIndices removeObjectAtIndex:indexOfObject];
        [_targets removeObject:targetToRemove];
        
        if ([targetToRemove respondsToSelector:@selector(releaseImageIndex:)]) {
            [targetToRemove releaseImageIndex:imageIndex];
        }
        
        DB_TRACE(-_verbose+2, "\n%s(%ld) rm %s", self.label.UTF8String, indexOfObject, [targetToRemove description].UTF8String);
    }
}

- (void)removeAllTargets {
    @synchronized (self) {
        [_targets removeAllObjects];
        [_targetIndices removeAllObjects];
        
        DB_TRACE(-_verbose+2, "\n%s rm all forward targets", self.label.UTF8String);
    }
}

@end
