//
//  MIOneInputConnectiveLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIOneInputConnectiveLayer.h"
#import "MPSImage+Extension.h"

@interface MIOneInputConnectiveLayer() {
}

@end

@implementation MIOneInputConnectiveLayer

- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape
{
    if (self = [super initWithOutputShape:outputShape]) {
        NSAssert(inputShape != NULL, @"Invalid input shape");
        NSAssert(outputShape != NULL, @"Invalid output shape");
        _firstInputShape = inputShape[0];
        
    }
    return self;
}

- (instancetype)init {
    if (self = [super init]) {
        
    }
    return self;
}

- (void)processTensorWithCommandBuffer:(id<MTLCommandBuffer>)cmdBuf {
    NSAssert(NO, @"Sub class should override the method.");
    [firstInputImage unlock];
}

#pragma mark - MILayerInput Delegate

- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)cmdBuf {
    [self processTensorWithCommandBuffer:cmdBuf];
}

- (void)setInputImage:(MITemporaryImage *)newInputImage atIndex:(NSInteger)imageIndex {
    NSParameterAssert(DataShapesTheSame(&_firstInputShape, [newInputImage shape]));
    firstInputImage = newInputImage;
    [firstInputImage lock];
}

MIOneInputConnectiveLayer *ConnectLinearLayers(NSArray<MIOneInputConnectiveLayer *> *layers) {
    MIOneInputConnectiveLayer *latest = layers.firstObject;
    for (int i = 1; i < layers.count; i++) {
        MIOneInputConnectiveLayer *newLayer = layers[i];
        [latest addTarget:newLayer];
        latest = newLayer;
    }
    return latest;
}

@end

