//
//  MIArithmeticLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/3.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIArithmeticLayer.h"
#import <MetalImage/MetalDevice.h>
#import "MITemporaryImageCache.h"

@implementation MIArithmeticLayer

+ (instancetype)additionArithmeticLayerWithDataShape:(DataShape *)dataShape {
    DataShape *inputShapes[2] = {dataShape, dataShape};
    MIArithmeticLayer *layer = [[MIArithmeticLayer alloc] initWithInputShapes1:inputShapes size:2 outputShape:dataShape];
    layer->dataShape = *dataShape;
    layer->arithmetic = [[MPSCNNAdd alloc] initWithDevice:[MetalDevice sharedMTLDevice]];
    layer->arithmetic.primaryScale = 1.0f;
    layer->arithmetic.secondaryScale = 1.0f;
    layer->arithmetic.bias = 0.0f;
    layer->arithmetic.primaryStrideInPixelsX = 1;
    layer->arithmetic.primaryStrideInPixelsY = 1;
    layer->arithmetic.primaryStrideInFeatureChannels = 1;
    return layer;
}

+ (instancetype)subtractionArithmeticLayerWithDataShape:(DataShape *)dataShape {
    DataShape *inputShapes[2] = {dataShape, dataShape};
    MIArithmeticLayer *layer = [[MIArithmeticLayer alloc] initWithInputShapes1:inputShapes size:2 outputShape:dataShape];
    layer->dataShape = *dataShape;
    layer->arithmetic = [[MPSCNNSubtract alloc] initWithDevice:[MetalDevice sharedMTLDevice]];
    layer->arithmetic.primaryScale = 1.0f;
    layer->arithmetic.secondaryScale = 1.0f;
    layer->arithmetic.bias = 0.0f;
    layer->arithmetic.primaryStrideInPixelsX = 1;
    layer->arithmetic.primaryStrideInPixelsY = 1;
    layer->arithmetic.primaryStrideInFeatureChannels = 1;
    return layer;
}

+ (instancetype)multiplicationArithmeticLayerWithDataShape:(DataShape *)dataShape {
    DataShape *inputShapes[2] = {dataShape, dataShape};
    MIArithmeticLayer *layer = [[MIArithmeticLayer alloc] initWithInputShapes1:inputShapes size:2 outputShape:dataShape];
    layer->dataShape = *dataShape;
    layer->arithmetic = [[MPSCNNMultiply alloc] initWithDevice:[MetalDevice sharedMTLDevice]];
    layer->arithmetic.primaryScale = 1.0f;
    layer->arithmetic.secondaryScale = 1.0f;
    layer->arithmetic.bias = 0.0f;
    layer->arithmetic.primaryStrideInPixelsX = 1;
    layer->arithmetic.primaryStrideInPixelsY = 1;
    layer->arithmetic.primaryStrideInFeatureChannels = 1;
    return layer;
}

+ (instancetype)divisionArithmeticLayerWithDataShape:(DataShape *)dataShape {
    DataShape *inputShapes[2] = {dataShape, dataShape};
    MIArithmeticLayer *layer = [[MIArithmeticLayer alloc] initWithInputShapes1:inputShapes size:2 outputShape:dataShape];
    layer->dataShape = *dataShape;
    layer->arithmetic = [[MPSCNNDivide alloc] initWithDevice:[MetalDevice sharedMTLDevice]];
    layer->arithmetic.primaryScale = 1.0f;
    layer->arithmetic.secondaryScale = 1.0f;
    layer->arithmetic.bias = 0.0f;
    layer->arithmetic.primaryStrideInPixelsX = 1;
    layer->arithmetic.primaryStrideInPixelsY = 1;
    layer->arithmetic.primaryStrideInFeatureChannels = 1;
    return layer;
}

- (void)setInputImage:(MITemporaryImage *)newInputImage atIndex:(NSInteger)imageIndex {
    NSAssert(DataShapesTheSame(newInputImage.shape, &dataShape), @"Invalid input tensor shape.");
    [super setInputImage:newInputImage atIndex:imageIndex];
    if (_secondaryImage) {
        [super setInputImage:(MITemporaryImage * _Nonnull)_secondaryImage atIndex:1];
    }
}

- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)cmdBuf {
    [super tempImageReadyAtIndex:imageIndex commandBuffer:cmdBuf];
    if (_secondaryImage) {
        [super tempImageReadyAtIndex:1 commandBuffer:cmdBuf];
    }
}

- (void)processTensorWithCommandBuffer:(id<MTLCommandBuffer>)cmdBuf {
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);
    
    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:cmdBuf];
    [_outputTempImage newTemporaryImageForCommandBuffer:cmdBuf];
    [arithmetic encodeToCommandBuffer:cmdBuf primaryImage:_inputs[@(0)].image secondaryImage:_inputs[@(1)].image destinationImage:_outputTempImage.image];
    [self removeCachedImages];
    [self notifyTargetsAboutNewTempImage:cmdBuf];
}

- (void)setDestinationFeatureChannelOffset:(NSInteger)channelOffset {
    [arithmetic setDestinationFeatureChannelOffset:channelOffset];
}

@end
