//
//  MetalLayerSwitch.m
//  MetalImage
//
//  Created by Feng Stone on 2019/8/1.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalLayerSwitch.h"

@implementation MetalLayerSwitch



- (ForwardTarget)activedTarget {
    NSArray *currentTargets = self.targets;
    NSUInteger index = _activedTargetIndex;
    if (index >= currentTargets.count) {
        return nil;
    }
    return currentTargets[index];
}

#pragma mark - MetalLayerInput delegate

- (DataShape *)dataShapeRef {
    return NULL;
}

- (void)setInputShape:(DataShape *)dataShape atIndex:(NSInteger)imageIndex {
    [[self activedTarget] setInputShape:dataShape atIndex:imageIndex];
}

- (void)setImage:(MetalTensor)newImage atIndex:(NSInteger)imageIndex {
    [[self activedTarget] setImage:newImage atIndex:imageIndex];
}

- (void)imageReadyOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer atIndex:(NSInteger)imageIndex {
    [[self activedTarget] imageReadyOnCommandBuffer:commandBuffer atIndex:imageIndex];
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
}

- (DataShape *)outputShapeRef {
    return [[self activedTarget] outputShapeRef];
}


@end
