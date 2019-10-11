//
//  MITwoInputsConnectiveLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/3.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MITwoInputsConnectiveLayer.h"

@implementation MITwoInputsConnectiveLayer

- (instancetype)initWithFirstInputShape:(DataShape *)firstInputShape
                       secondInputShape:(DataShape *)secondInputShape
                            outputShape:(DataShape *)outputShape {
    if (self = [super initWithInputShape:firstInputShape outputShape:outputShape]) {
        _secondInputShape = *secondInputShape;
        _firstDataFlags.targetHasBeenSet = NO;
        _firstDataFlags.hasReceivedFrame = NO;
        _secondDataFlags.targetHasBeenSet = NO;
        _secondDataFlags.hasReceivedFrame = NO;
    }
    return self;
}

#pragma mark - MILayerInput Delegate

- (NSInteger)nextAvailableTextureIndex {
    if (_firstDataFlags.targetHasBeenSet) {
        return 1;
    }
    else {
        return 0;
    }
}

- (void)reserveTextureIndex:(NSInteger)index {
    if (index == 0) {
        _firstDataFlags.targetHasBeenSet = YES;
    }
}

- (void)releaseTextureIndex:(NSInteger)index {
    if (index == 0) {
        _firstDataFlags.targetHasBeenSet = NO;
    }
}

- (void)setInputImage:(MITemporaryImage *)newInputImage atIndex:(NSInteger)imageIndex {
    if (imageIndex == 0) {
        firstInputImage = newInputImage;
        [firstInputImage lock];
    }
    else if (imageIndex == 1) {
        secondInputImage = newInputImage;
        [secondInputImage lock];
    }
    else {
        NSAssert(NO, @"Invalid input image index.");
    }
}

- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)cmdBuf {
    if (_firstDataFlags.hasReceivedFrame && _secondDataFlags.hasReceivedFrame) {
        return;
    }
    
    if (imageIndex == 0)
    {
        _firstDataFlags.hasReceivedFrame = YES;
    }
    else
    {
        _secondDataFlags.hasReceivedFrame = YES;
    }
    
    if (_firstDataFlags.hasReceivedFrame && _secondDataFlags.hasReceivedFrame)
    {
        [self processTensorWithCommandBuffer:cmdBuf];
        _firstDataFlags.hasReceivedFrame = NO;
        _secondDataFlags.hasReceivedFrame = NO;
    }
}


@end
