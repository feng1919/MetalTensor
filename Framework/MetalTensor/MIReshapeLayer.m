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
    BackwardTarget _backwardTarget;
}

@end

@implementation MIReshapeLayer

- (instancetype)init {
    NSAssert(NO, @"The reshape layer's output shape must be sepecified explicitly. Use -initWithInputShape:outputShape:.");
    return nil;
}

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
        
        NSAssert(Product(&_inputShapes[0]) == Product(&_outputShape), @"Reshape can not be cast, because the product of input shape is not equal to the product of output shape.");
        
        DB_TRACE(-_verbose+2, "\n%s init %s --> %s", self.labelUTF8, NSStringFromDataShape(inputShape).UTF8String, NSStringFromDataShape(outputShape).UTF8String);
    }
    return self;
}

#pragma mark - override
- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    _reshape = [[MPSNNReshape alloc] initWithDevice:device];
}

#pragma mark - MTTensorForward Delegate
- (void)setInputShape:(DataShape *)dataShape atIndex:(NSInteger)imageIndex {
    [super setInputShape:dataShape atIndex:imageIndex];
    NSLog(@"The reshape layer %@ need an output shape specified explicitly, the input shape is changed.", self.label);
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+3, "\n%s encoding...", self.labelUTF8);
    
    NSAssert(_operation, @"The computing operation has not been initialized.");
    NSAssert(_inputImages.count > 0, @"There is no input image received.");
    
    _image = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape commandBuffer:commandBuffer];
    _image.source = self;
    
    MetalTensor sourceTensor = _inputImages[@(0)];
    _backwardTarget = sourceTensor.source;
    
    [_reshape encodeToCommandBuffer:commandBuffer
                          sourceImage:sourceTensor.content
                     destinationImage:_image.content];
    
    [self removeCachedImages];

#if DEBUG
    if (self.dumpResult) {
        [self saveTensor:_image onCommandBuffer:commandBuffer];
    }
#endif
}

#pragma mark - MTTensorBackward Delegate
- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    //  Reshape the gradient from backward node to forward input tensor shape.
    MetalTensor destinationImage = [[MTTensorCache sharedCache] fetchTensorWithShape:&_inputShapes[0]
                                                                       commandBuffer:commandBuffer];
    NSAssert(_backwardTarget, @"Invalid backward target...");
    
    [_reshape encodeToCommandBuffer:commandBuffer
                        sourceImage:_gradient.content
                   destinationImage:destinationImage.content];
    
    [self removeGradient];
    
    if (self.stopGradient) {
        [self.blit encodeToCommandBuffer:commandBuffer
                             sourceImage:destinationImage.content
                        destinationImage:self.savedGradients.content];
        [destinationImage unlock];
    }
    else {
    
        [_backwardTarget setGradient:destinationImage forwardTarget:self];
        [destinationImage unlock];
        [_backwardTarget gradientReadyOnCommandBuffer:commandBuffer forwardTarget:self];
    }
}

MIReshapeLayer *MakeReshapeLayer(DataShape *inputShape, DataShape *outputShape) {
    return [[MIReshapeLayer alloc] initWithInputShape:inputShape outputShape:outputShape];
}

@end
