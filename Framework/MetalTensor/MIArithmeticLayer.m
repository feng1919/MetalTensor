//
//  MIArithmeticLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/3.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIArithmeticLayer.h"
#import "MTTensorCache.h"

@implementation MIArithmeticLayer {
    
@protected
    MPSCNNArithmetic *_arithmetic;
    MPSCNNArithmeticGradient *_primaryGradientOperation;
    MPSCNNArithmeticGradient *_secondaryGradientOperation;
}

+ (instancetype)arithmeticLayerWithDataShape:(DataShape *)dataShape {
    DataShape *inputShapes[2] = {dataShape, dataShape};
    return [[MIArithmeticLayer alloc] initWithInputShapes1:inputShapes size:2];
}

- (void)initializeArithmetic {
    
    NSParameterAssert(_arithmeticType.length > 0);
    
    if (!_device) {
        return;
    }
    
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
    
    if (_needBackward) {
        Class gradientClass = [MIArithmeticLayer arithmeticGradientWithType:_arithmeticType];
        _primaryGradientOperation = [[gradientClass alloc] initWithDevice:_device isSecondarySourceFilter:NO];
        _secondaryGradientOperation = [[gradientClass alloc] initWithDevice:_device isSecondarySourceFilter:YES];
    }
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

- (void)setImage:(MetalTensor)newImage atIndex:(NSInteger)imageIndex {
    NSAssert(DataShapesTheSame(newImage.shape, &_inputShapes[0]), @"Invalid input tensor shape.");
    [super setImage:newImage atIndex:imageIndex];
    if (_secondaryImage) {
        [super setImage:(MetalTensor  _Nonnull)_secondaryImage atIndex:1];
    }
}

- (void)imageReadyAtIndex:(NSInteger)imageIndex onCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    [super imageReadyAtIndex:imageIndex onCommandBuffer:commandBuffer];
    if (_secondaryImage) {
        [super imageReadyAtIndex:1 onCommandBuffer:commandBuffer];
    }
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);
    
    _image = [[MTTensorCache sharedCache] fetchTensorWithShape:&_dataShape source:self commandBuffer:commandBuffer];
    [_image newContentOnCommandBuffer:commandBuffer];
    if (_needBackward) {
        
        _state = [_arithmetic temporaryResultStateForCommandBuffer:commandBuffer
                                                      primaryImage:_inputImages[@(0)].content
                                                    secondaryImage:_inputImages[@(1)].content
                                                      sourceStates:nil
                                                  destinationImage:_image.content];
        
        [_arithmetic encodeToCommandBuffer:commandBuffer
                              primaryImage:_inputImages[@(0)].content
                            secondaryImage:_inputImages[@(1)].content
                          destinationState:(MPSCNNArithmeticGradientState *)_state
                          destinationImage:_image.content];
    }
    else {
        
        [_arithmetic encodeToCommandBuffer:commandBuffer
                              primaryImage:_inputImages[@(0)].content
                            secondaryImage:_inputImages[@(1)].content
                          destinationImage:_image.content];
        [self removeCachedImages];
    }
}

- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    MetalTensor primaryTensor = _inputImages[@(0)];
    MetalTensor secondaryTensor = _inputImages[@(1)];
    BackwardTarget primaryBackward = primaryTensor.source;
    BackwardTarget secondaryBackward = secondaryTensor.source;
    
    MetalTensor primaryGradient = [[MTTensorCache sharedCache] fetchTensorWithShape:&_dataShape source:nil commandBuffer:commandBuffer];
    [_primaryGradientOperation encodeToCommandBuffer:commandBuffer
                                      sourceGradient:_gradient.content
                                         sourceImage:primaryTensor.content
                                       gradientState:_state
                                 destinationGradient:primaryGradient.content];
    [primaryBackward setGradient:primaryGradient forwardTarget:self];
    [primaryGradient unlock];
    
    if (secondaryBackward) {
        MetalTensor secondaryGradient = [[MTTensorCache sharedCache] fetchTensorWithShape:&_dataShape source:nil commandBuffer:commandBuffer];
        [_secondaryGradientOperation encodeToCommandBuffer:commandBuffer
                                            sourceGradient:_gradient.content
                                               sourceImage:secondaryTensor.content
                                             gradientState:_state
                                       destinationGradient:secondaryGradient.content];
        [secondaryBackward setGradient:secondaryGradient forwardTarget:self];
        [secondaryGradient unlock];
    }
    
    [self removeState];
    [self removeCachedImages];
    [self removeCachedGradients];
    
    [primaryBackward gradientReadyFromForwardTarget:self onCommandBuffer:commandBuffer];
    [secondaryBackward gradientReadyFromForwardTarget:self onCommandBuffer:commandBuffer];
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

+ (Class)arithmeticGradientWithType:(NSString *)arithmetic {
    if ([arithmetic isEqualToString:@"addition"]) {
        return [MPSCNNAddGradient class];
    }
    if ([arithmetic isEqualToString:@"divide"]) {
//        return [MPSCNNDivide class];
    }
    if ([arithmetic isEqualToString:@"subtract"]) {
        return [MPSCNNSubtractGradient class];
    }
    if ([arithmetic isEqualToString:@"multiply"]) {
        return [MPSCNNMultiplyGradient class];
    }
    assert(0);
    return nil;
}

@end
