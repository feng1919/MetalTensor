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

- (id)initWithOutputShape:(DataShape *)outputShape
{
    NSParameterAssert(outputShape!=NULL);
    if (!(self = [super init]))
    {
        return nil;
    }
    
//#ifdef DEBUG
    _verbose = _METAL_VERBOSE;
//#endif
    _outputShape = outputShape[0];
    _targets = [[NSMutableArray alloc] init];
    _targetTempImageIndices = [[NSMutableArray alloc] init];
    
    return self;
}

- (instancetype)init {
    if (!(self = [super init]))
    {
        return nil;
    }
    
//#ifdef DEBUG
    _verbose = _METAL_VERBOSE;
//#endif
    _targets = [[NSMutableArray alloc] init];
    _targetTempImageIndices = [[NSMutableArray alloc] init];
    
    return self;
}

- (DataShape *)outputShapeRef {
    return &_outputShape;
}

- (void)dealloc
{
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

#pragma mark - Managing targets

- (MITemporaryImage *)outputTempImage {
    return _outputTempImage;
}

- (void)removeOutputTempImage {
    
    DB_TRACE(-_verbose+2, "\n%s rm %s",
             self.labelUTF8,
             NSStringFromDataShape(_outputTempImage.shape).UTF8String);
    
    [_outputTempImage unlock];
    _outputTempImage = nil;
    
}

- (void)setOutputTempImageToTargets {
    for (id<MetalTensorInput> currentTarget in _targets) {
        NSInteger indexOfObject = [_targets indexOfObject:currentTarget];
        NSInteger tempImageIndex = [_targetTempImageIndices[indexOfObject] integerValue];
        [currentTarget setInputImage:_outputTempImage atIndex:tempImageIndex];
        
        DB_TRACE(-_verbose+1, "\n%s ---%s---> %s(%ld)",
                 self.labelUTF8,
                 NSStringFromDataShape(_outputTempImage.shape).UTF8String,
                 [currentTarget description].UTF8String,
                 tempImageIndex);
    }
}

- (void)notifyTargetsAboutNewTempImage:(id<MTLCommandBuffer>)cmdBuf {

    [self setOutputTempImageToTargets];
    [self removeOutputTempImage];
    
    for (id<MetalTensorInput> currentTarget in _targets)
    {
        NSInteger indexOfObject = [_targets indexOfObject:currentTarget];
        NSInteger tempImageIndex = [[_targetTempImageIndices objectAtIndex:indexOfObject] integerValue];
        [currentTarget tempImageReadyAtIndex:tempImageIndex commandBuffer:cmdBuf];
    }
}

- (NSArray*)targets
{
    @synchronized (self) {
        return [NSArray arrayWithArray:_targets];
    }
}

- (void)addTarget:(id<MetalTensorInput>)newTarget
{
    @synchronized (self) {
        NSInteger nextAvailableTempImageIndex = 0;
        if ([newTarget respondsToSelector:@selector(nextAvailableTempImageIndex)]) {
            nextAvailableTempImageIndex = [newTarget nextAvailableTempImageIndex];
        }
        
        [self addTarget:newTarget atTempImageIndex:nextAvailableTempImageIndex];
    }
}

- (void)addTarget:(id<MetalTensorInput>)newTarget atTempImageIndex:(NSInteger)imageIndex
{
    @synchronized (self) {
        if ([_targets containsObject:newTarget]) {
            return;
        }
        
        [_targets addObject:newTarget];
        [_targetTempImageIndices addObject:@(imageIndex)];
        
        if ([newTarget respondsToSelector:@selector(reserveTempImageIndex:)]) {
            [newTarget reserveTempImageIndex:imageIndex];
        }
        
        DB_TRACE(-_verbose+2, "\n%s add %s at %ld", self.labelUTF8, [newTarget description].UTF8String, imageIndex);
    }
}

- (void)removeTarget:(id<MetalTensorInput>)targetToRemove
{
    @synchronized (self) {
        if(![_targets containsObject:targetToRemove])
        {
            return;
        }
        
        NSInteger indexOfObject = [_targets indexOfObject:targetToRemove];
        NSInteger tempImageIndexOfTarget = [[_targetTempImageIndices objectAtIndex:indexOfObject] integerValue];
        
        [_targetTempImageIndices removeObjectAtIndex:indexOfObject];
        [_targets removeObject:targetToRemove];
        
        if ([targetToRemove respondsToSelector:@selector(releaseTempImageIndex:)]) {
            [targetToRemove releaseTempImageIndex:tempImageIndexOfTarget];
        }
        
        DB_TRACE(-_verbose+2, "\n%s(%ld) rm %s", self.label, indexOfObject, [targetToRemove description].UTF8String);
    }
}

- (void)removeAllTargets
{
    @synchronized (self) {
        [_targets removeAllObjects];
        [_targetTempImageIndices removeAllObjects];
        
        DB_TRACE(-_verbose+2, "\n%s rm all targets", self.label);
    }
}

@end

