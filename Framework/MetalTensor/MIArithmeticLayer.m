//
//  MIArithmeticLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/3.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIArithmeticLayer.h"
#import "MITemporaryImageCache.h"

@implementation MIArithmeticLayer

+ (instancetype)arithmeticLayerWithDataShape:(DataShape *)dataShape {
    DataShape *inputShapes[2] = {dataShape, dataShape};
    return [[MIArithmeticLayer alloc] initWithInputShapes1:inputShapes size:2];
}

- (void)initializeArithmetic {
    
    NSParameterAssert(_arithmeticType.length > 0);
    NSParameterAssert(_device);
    
    Class arithmeticClass = [MIArithmeticLayer arithmeticWithType:_arithmeticType];
    _arithmetic = [[arithmeticClass alloc] initWithDevice:_device];
    _arithmetic.primaryScale = 1.0f;
    _arithmetic.bias = 0.0f;
    _arithmetic.primaryStrideInPixelsX = 1;
    _arithmetic.primaryStrideInPixelsY = 1;
    _arithmetic.primaryStrideInFeatureChannels = 1;
    _arithmetic.secondaryScale = 1.0f;
    _arithmetic.secondaryStrideInPixelsX = 1;
    _arithmetic.secondaryStrideInPixelsY = 1;
    _arithmetic.secondaryStrideInFeatureChannels = 1;
    _arithmetic.destinationFeatureChannelOffset = _channelOffset;
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    [self initializeArithmetic];
}

- (void)setChannelOffset:(NSInteger)channelOffset {
    _channelOffset = channelOffset;
    _arithmetic.destinationFeatureChannelOffset = channelOffset;
}

- (void)setArithmeticType:(NSString *)arithmeticType {
    if (![_arithmeticType isEqualToString:arithmeticType]) {
        _arithmeticType = arithmeticType;
        
        if (_device) {
            [self initializeArithmetic];
        }
    }
}

- (void)setInputImage:(MITemporaryImage *)newInputImage atIndex:(NSInteger)imageIndex {
    NSAssert(DataShapesTheSame(newInputImage.shape, &_inputShapes[0]), @"Invalid input tensor shape.");
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
    [_arithmetic encodeToCommandBuffer:cmdBuf
                          primaryImage:_inputs[@(0)].image
                        secondaryImage:_inputs[@(1)].image
                      destinationImage:_outputTempImage.image];
    [self removeCachedImages];
    [self notifyTargetsAboutNewTempImage:cmdBuf];
}

+ (Class)arithmeticWithType:(NSString *)arithmetic {
    if ([arithmetic isEqualToString:@"addition"]) {
        return [MPSCNNAdd class];
    }
    if ([arithmetic isEqualToString:@"divide"]) {
        return [MPSCNNDivide class];
    }
    if ([arithmetic isEqualToString:@"subtract"]) {
        return [MPSCNNSubtract class];
    }
    if ([arithmetic isEqualToString:@"multiply"]) {
        return [MPSCNNMultiply class];
    }
    assert(0);
    return nil;
}

@end
