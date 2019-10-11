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

- (void)setInputImage:(MITemporaryImage *)newInputImage atIndex:(NSInteger)imageIndex {
    [_targets[_activedTargetIndex] setInputImage:newInputImage atIndex:imageIndex];
}

- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)cmdBuf {
    [_targets[_activedTargetIndex] tempImageReadyAtIndex:imageIndex commandBuffer:cmdBuf];
}

- (void)processTensorWithCommandBuffer:(id<MTLCommandBuffer>)cmdBuf {
    
}

@end
