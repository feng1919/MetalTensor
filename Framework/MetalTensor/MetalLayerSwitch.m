//
//  MetalLayerSwitch.m
//  MetalImage
//
//  Created by Feng Stone on 2019/8/1.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalLayerSwitch.h"

@implementation MetalLayerSwitch


#pragma mark - MetalLayerInput delegate

- (DataShape *)dataShapeRef {
    return NULL;
}

- (void)setImage:(MetalTensor)newImage atIndex:(NSInteger)imageIndex {
    [_targets[_activedTargetIndex] setImage:newImage atIndex:imageIndex];
}

- (void)imageReadyAtIndex:(NSInteger)imageIndex onCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    [_targets[_activedTargetIndex] imageReadyAtIndex:imageIndex onCommandBuffer:commandBuffer];
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
}

- (DataShape *)outputShapeRef {
    return [_targets[_activedTargetIndex] outputShapeRef];
}


@end
