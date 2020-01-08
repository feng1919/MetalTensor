//
//  MIReshapeLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/23.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIReshapeLayer.h"
#import "MTTensor.h"
#import "MTTensorCache.h"

@interface MIReshapeLayer() {
    MPSNNReshape *_reshape;
}

@end

@implementation MIReshapeLayer

- (instancetype)initWithInputShape:(DataShape *)inputShape {
    NSAssert(NO, @"The reshape layer's output shape must be sepecified explicitly. Use -initWithInputShape:outputShape:.");
    return nil;
}

- (instancetype)initWithInputShapes:(DataShape *)inputShapes size:(int)size {
    NSAssert(NO, @"The reshape layer's output shape must be sepecified explicitly. Use -initWithInputShape:outputShape:.");
    return nil;
}

- (instancetype)initWithInputShapes1:(DataShape * _Nonnull *)inputShapes size:(int)size {
    NSAssert(NO, @"The reshape layer's output shape must be sepecified explicitly. Use -initWithInputShape:outputShape:.");
    return nil;
}

- (instancetype)initWithInputShape:(DataShape *)inputShape outputShape:(DataShape *)outputShape {
    if (self = [super initWithInputShapes:inputShape size:1]) {
        
        _outputShape = *outputShape;
        
        NSAssert(ProductOfDataShape(&_inputShapes[0]) == ProductOfDataShape(&_outputShape), @"Reshape can not be cast, because the product of input shape is not equal to the product of output shape.");
        
        DB_TRACE(-_verbose+2, "\n%s init %s --> %s", self.labelUTF8, NSStringFromDataShape(inputShape).UTF8String, NSStringFromDataShape(outputShape).UTF8String);
    }
    return self;
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    _reshape = [[MPSNNReshape alloc] initWithDevice:device];
    
    _operation = _reshape;
}

- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    //  Reshape the gradient from backward node to forward input tensor shape.
    MetalTensor sourceTensor = _inputImages[@(0)];
    [_reshape encodeToCommandBuffer:commandBuffer sourceImage:_gradient.content destinationImage:sourceTensor.content];
    
    [sourceTensor.source setGradient:sourceTensor forwardTarget:self];
    [self removeCachedImages];
    [self removeGradient];
    [sourceTensor.source gradientReadyFromForwardTarget:self onCommandBuffer:commandBuffer];
    
}

MIReshapeLayer *MakeReshapeLayer(DataShape *inputShape, DataShape *outputShape) {
    return [[MIReshapeLayer alloc] initWithInputShape:inputShape outputShape:outputShape];
}

@end
