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

#pragma mark - override

- (void)initialize {
    _bias = 0.0f;
    _primaryScale = 1.0f;
    _secondaryScale = 1.0f;
    _channelOffset = 0;
    _primaryStrides = MTLInt3Make(1, 1, 1);
    _secondaryStrides = MTLInt3Make(1, 1, 1);
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    [self initializeArithmetic];
}

#pragma mark - public

- (void)initializeArithmetic {
    
    NSParameterAssert(_arithmeticType.length > 0);
    
    if (!_device) {
        return;
    }
    
    Class arithmeticClass = [MIArithmeticLayer arithmeticWithType:_arithmeticType];
    _arithmetic = [[arithmeticClass alloc] initWithDevice:_device];
    [self updateArithmeticParameters];
    
    if (_needBackward) {
        Class gradientClass = [MIArithmeticLayer arithmeticGradientWithType:_arithmeticType];
        _primaryGradientOperation = [[gradientClass alloc] initWithDevice:_device isSecondarySourceFilter:NO];
        _secondaryGradientOperation = [[gradientClass alloc] initWithDevice:_device isSecondarySourceFilter:YES];
    }
}

- (void)updateArithmeticParameters {
    _arithmetic.primaryScale = _primaryScale;
    _arithmetic.secondaryScale = _secondaryScale;
    _arithmetic.bias = _bias;
    _arithmetic.primaryStrideInPixelsX = _primaryStrides.x;
    _arithmetic.primaryStrideInPixelsY = _primaryStrides.y;
    _arithmetic.primaryStrideInFeatureChannels = _primaryStrides.z;
    _arithmetic.secondaryStrideInPixelsX = _secondaryStrides.x;
    _arithmetic.secondaryStrideInPixelsY = _secondaryStrides.y;
    _arithmetic.secondaryStrideInFeatureChannels = _secondaryStrides.z;
    _arithmetic.destinationFeatureChannelOffset = _channelOffset;
}

- (void)setPrimaryScale:(float)primaryScale {
    _primaryScale = primaryScale;
    _arithmetic.primaryScale = primaryScale;
}

- (void)setSecondaryScale:(float)secondaryScale {
    _secondaryScale = secondaryScale;
    _arithmetic.secondaryScale = secondaryScale;
}

- (void)setBias:(float)bias {
    _bias = bias;
    _arithmetic.bias = bias;
}

- (void)setPrimaryStrides:(MTLInt3)primaryStrides {
    _primaryStrides = primaryStrides;
    _arithmetic.primaryStrideInPixelsX = primaryStrides.x;
    _arithmetic.primaryStrideInPixelsY = primaryStrides.y;
    _arithmetic.primaryStrideInFeatureChannels = primaryStrides.z;
}

- (void)setSecondaryStrides:(MTLInt3)secondaryStrides {
    _secondaryStrides = secondaryStrides;
    _arithmetic.secondaryStrideInPixelsX = secondaryStrides.x;
    _arithmetic.secondaryStrideInPixelsY = secondaryStrides.y;
    _arithmetic.secondaryStrideInFeatureChannels = secondaryStrides.z;
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

- (void)setStopGradient:(BOOL)stopGradient {
    NSAssert(NO, @"The arithmetic layer does not support stop gradient.");
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

#pragma mark - MTTensorForward delegate

- (void)setImage:(MetalTensor)newImage atIndex:(NSInteger)imageIndex {
    NSAssert(DataShapesTheSame(newImage.shape, &_inputShapes[0]), @"Invalid input tensor shape.");
    [super setImage:newImage atIndex:imageIndex];
    if (_secondaryImage) {
        [super setImage:(MetalTensor  _Nonnull)_secondaryImage atIndex:1];
    }
}

- (void)imageReadyOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer atIndex:(NSInteger)imageIndex {
    [super imageReadyOnCommandBuffer:commandBuffer atIndex:imageIndex];
    if (_secondaryImage) {
        [super imageReadyOnCommandBuffer:commandBuffer atIndex:1];
    }
}

- (void)reserveImageIndex:(NSInteger)index {
    [super reserveImageIndex:index];
    if (_secondaryImage) {
        [super reserveImageIndex:1];
    }
}

- (void)releaseImageIndex:(NSInteger)index {
    [super releaseImageIndex:index];
    if (_secondaryImage) {
        [super releaseImageIndex:1];
    }
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+2, "\n%s forward encoding...", self.labelUTF8);
    
    _image = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape commandBuffer:commandBuffer];
    _image.source = self;
    
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

#pragma mark - MTTensorBackward delegate

- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    DB_TRACE(-_verbose+2, "\n%s backward encoding...", self.labelUTF8);
    
    MetalTensor primaryTensor = _inputImages[@(0)];
    MetalTensor secondaryTensor = _inputImages[@(1)];
    BackwardTarget primaryBackward = primaryTensor.source;
    NSAssert(primaryBackward, @"Invalid primary backward target...");
    BackwardTarget secondaryBackward = secondaryTensor.source;
    NSAssert(secondaryTensor, @"Invalid secondary backward target...");
    
    MetalTensor primaryGradient = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape
                                                                      commandBuffer:commandBuffer];
    [_primaryGradientOperation encodeToCommandBuffer:commandBuffer
                                      sourceGradient:_gradient.content
                                         sourceImage:primaryTensor.content
                                       gradientState:_state
                                 destinationGradient:primaryGradient.content];
    [primaryBackward setGradient:primaryGradient forwardTarget:self];
    [primaryGradient unlock];
    
    if (secondaryBackward) {
        MetalTensor secondaryGradient = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape
                                                                            commandBuffer:commandBuffer];
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
    
    [primaryBackward gradientReadyOnCommandBuffer:commandBuffer forwardTarget:self];
    [secondaryBackward gradientReadyOnCommandBuffer:commandBuffer forwardTarget:self];
    
}

@end
