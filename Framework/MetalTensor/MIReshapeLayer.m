//
//  MIReshapeLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/23.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIReshapeLayer.h"
#import "MITemporaryImage.h"
#import "MITemporaryImageCache.h"

@interface MIReshapeLayer() {
    MPSNNReshape *_reshape;
}

@end

@implementation MIReshapeLayer

- (instancetype)initWithInputShape:(DataShape *)inputShape outputShape:(DataShape *)outputShape {
    if (self = [super initWithInputShape:inputShape]) {
        
        _outputShape = *outputShape;
        
        NSAssert(ProductOfDataShape(&_inputShapes[0]) == ProductOfDataShape(&_outputShape), @"Reshape can not be cast, because the product of input shape is not equal to the product of output shape.");
        
        DB_TRACE(-_verbose+2, "\n%s init %s --> %s", self.labelUTF8, NSStringFromDataShape(inputShape).UTF8String, NSStringFromDataShape(outputShape).UTF8String);
    }
    return self;
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    _reshape = [[MPSNNReshape alloc] initWithDevice:device];
}

- (void)processTensorWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);
    
    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:commandBuffer];
    [_outputTempImage newTemporaryImageForCommandBuffer:commandBuffer];
    [_reshape encodeToCommandBuffer:commandBuffer
                        sourceImage:_inputs[@(0)].image
                   destinationImage:_outputTempImage.image];
    
    [self removeCachedImages];
    
    [self notifyTargetsAboutNewTempImage:commandBuffer];
}

MIReshapeLayer *MakeReshapeLayer(DataShape *inputShape, DataShape *outputShape) {
    return [[MIReshapeLayer alloc] initWithInputShape:inputShape outputShape:outputShape];
}

@end
