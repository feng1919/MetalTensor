//
//  MetalTensorLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/7/4.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"
#import "MTTensorCache.h"
#include "numpy.h"
#import "MPSImage+Extension.h"

static MPSCNNAdd *_reduceSumOperation = nil;

@implementation MetalTensorLayer

- (instancetype)init {
    NSAssert(NO, @"Invalid initialize method.");
    return nil;
}

- (instancetype)initWithInputShapes1:(DataShape *_Nonnull*_Nonnull)inputShapes size:(int)size {
    
    if (self = [super init]) {
        
        NSAssert(size > 0, @"Invalid inputs number, if layer don't have any input, use MetalTensorNode.");
        int bitCount = sizeof(unsigned long long)<<3;
        NSAssert(size <= bitCount, @"Too many input tensors, support %d tensors at most.", bitCount);

        _reservedTargetFlags = 0x0ULL;
        _receivedImageFlags = 0x0ULL;
        _inputShapes = malloc(size * sizeof(DataShape));
        for (int i = 0; i < size; i++) {
            _inputShapes[i] = inputShapes[i][0];
        }
        
        _numOfImages = size;
        
        _inputImages = [NSMutableDictionary dictionaryWithCapacity:size];
        _inputGradients = [NSMutableArray array];
        
#ifdef DEBUG
        NSMutableString *string = [NSMutableString string];
        for (int i = 0; i < size; i++) {
            [string appendString:NSStringFromDataShape(inputShapes[i])];
            if (i != size-1) {
                [string appendString:@", "];
            }
        }
        
        DB_TRACE(-_verbose+2, "\n%s init [%s]", self.labelUTF8, string.UTF8String);
#endif
        
        [self initialize];
    }
    return self;
}

- (instancetype)initWithInputShapes:(DataShape *_Nonnull)inputShapes size:(int)size {
    
    if (self = [super init]) {
        
        NSAssert(size > 0, @"Invalid inputs number, if layer don't have any input, use MetalTensorNode.");
        int bitCount = sizeof(unsigned long long)<<3;
        NSAssert(size <= bitCount, @"Too many input tensors, support %d tensors at most.", bitCount);

        _reservedTargetFlags = 0x0ULL;
        _receivedImageFlags = 0x0ULL;
        _inputShapes = malloc(size * sizeof(DataShape));
        npmemcpy(_inputShapes, inputShapes, size*sizeof(DataShape));
        
        _numOfImages = size;
        
        _inputImages = [NSMutableDictionary dictionaryWithCapacity:size];
        _inputGradients = [NSMutableArray array];
#if DEBUG
        NSMutableString *string = [NSMutableString string];
        for (int i = 0; i < size; i++) {
            [string appendString:NSStringFromDataShape(&inputShapes[i])];
            if (i != size-1) {
                [string appendString:@", "];
            }
        }
        
        DB_TRACE(-_verbose+2, "\n%s init [%s]", self.labelUTF8, string.UTF8String);
#endif
        
        [self initialize];
    }
    return self;
}

- (instancetype)initWithInputShape:(DataShape *)inputShape {
    return [self initWithInputShapes:inputShape size:1];
}

- (void)initialize {
}

- (void)dealloc {
    free(_inputShapes);
    _inputShapes = NULL;
}

#pragma mark - GET

- (DataShape *)inputShapes {
    return _inputShapes;
}

- (int)numOfImages {
    return _numOfImages;
}

- (int)numOfGradients {
    return (int)_targets.count;
}

#pragma mark - Compile the layer

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    NSParameterAssert(_numOfImages > 0);
    if (ProductOfDataShape(&_outputShape) == 0) {
        _outputShape = _inputShapes[0];
    }
}

#pragma mark - Forward Processing

- (DataShape *)dataShapeRef {
    return &_outputShape;
}

- (void)setImage:(MetalTensor)newImage atIndex:(NSInteger)imageIndex {
    NSAssert(imageIndex < _numOfImages, @"Invalid input image index.");
    NSAssert(DataShapesTheSame(&_inputShapes[imageIndex], [newImage shape]), @"The input image's shape is not identical to the inputShapes[%d]", (int)imageIndex);
    _inputImages[@(imageIndex)] = newImage;
    [newImage lock];
    
    DB_TRACE(-_verbose+2, "\n%s(%ld) <-- %s", self.labelUTF8, imageIndex, NSStringFromDataShape(newImage.shape).UTF8String);
}

- (void)imageReadyAtIndex:(NSInteger)imageIndex onCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    if ([self isAllImagesReceived]) {
        return;
    }
    
    [self receiveImageAtIndex:imageIndex];
    
    if ([self isAllImagesReceived]) {
        [self processImagesOnCommandBuffer:commandBuffer];
        [self notifyTargetsAboutNewImageOnCommandBuffer:commandBuffer];
        [self resetImagesReceivedFlags];
    }
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+3, "\n%s encoding...", self.labelUTF8);
    
    _image = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape source:self commandBuffer:commandBuffer];
    [_image newContentOnCommandBuffer:commandBuffer];
    
    MetalTensor sourceTensor = _inputImages[@(0)];
    if (_needBackward) {
        //  Save the forward operation state.
        _state = [_operation temporaryResultStateForCommandBuffer:commandBuffer
                                                      sourceImage:sourceTensor.content
                                                     sourceStates:nil
                                                 destinationImage:_image.content];
        [_operation encodeToCommandBuffer:commandBuffer
                              sourceImage:sourceTensor.content
                         destinationState:_state
                         destinationImage:_image.content];
        
        //  In backward mode, we need the source images to compute gradients,
        //  so we keep the source images alive.
    }
    else {
        [_operation encodeToCommandBuffer:commandBuffer
                              sourceImage:sourceTensor.content
                         destinationImage:_image.content];
        
        //  Release the source images.
        [self removeCachedImages];
    }
}

- (void)removeCachedImages {
    [_inputImages.allValues makeObjectsPerformSelector:@selector(unlock)];
    [_inputImages removeAllObjects];
    
    DB_TRACE(-_verbose+3, "\n%s rm all inputs", self.labelUTF8);
}

- (void)removeImage {
    if (_image) {
        DB_TRACE(-_verbose+2, "\n%s rm %s",
                 self.labelUTF8,
                 NSStringFromDataShape(_image.shape).UTF8String);
        
        [_image unlock];
        _image = nil;
    }
}

- (void)setImageToTargets {
    for (ForwardTarget currentTarget in _targets) {
        NSInteger indexOfObject = [_targets indexOfObject:currentTarget];
        NSInteger index = [_targetIndices[indexOfObject] integerValue];
        [currentTarget setImage:_image atIndex:index];
        
        DB_TRACE(-_verbose+1, "\n%s ---%s---> %s(%ld)",
                 self.labelUTF8,
                 NSStringFromDataShape(_image.shape).UTF8String,
                 [currentTarget description].UTF8String,
                 index);
    }
}

- (void)notifyTargetsAboutNewImageOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {

    [self setImageToTargets];
    [self removeImage];
    
    for (ForwardTarget currentTarget in _targets)
    {
        NSInteger indexOfObject = [_targets indexOfObject:currentTarget];
        NSInteger imageIndex = [[_targetIndices objectAtIndex:indexOfObject] integerValue];
        [currentTarget imageReadyAtIndex:imageIndex onCommandBuffer:commandBuffer];
    }
}

#pragma mark - Backward Processing
- (void)setGradient:(MetalTensor)newGradient forwardTarget:(ForwardTarget)target{
    NSAssert(DataShapesTheSame(&_outputShape, [newGradient shape]), @"The input gradient's shape is not identical to the output shape.");
    [_inputGradients addObject:newGradient];
    [newGradient lock];
    
    DB_TRACE(-_verbose+2, "\n%s <-- %s", self.labelUTF8, NSStringFromDataShape(newGradient.shape).UTF8String);
}

- (void)gradientReadyFromForwardTarget:(ForwardTarget)target onCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    if ([self isAllGradientsReceived]) {
        [self reduceSumBatchGradientsOnCommandBuffer:commandBuffer];
        [self processGradientsOnCommandBuffer:commandBuffer];
    }
}

- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    MetalTensor sourceTensor = _inputImages[@(0)];
    NSAssert(sourceTensor.source, @"Invalid backward connection...");
    
    MetalTensor destinationGradient = [[MTTensorCache sharedCache] fetchTensorWithShape:sourceTensor.shape
                                                                                 source:nil
                                                                          commandBuffer:commandBuffer];
    [_gradientOp encodeToCommandBuffer:commandBuffer
                        sourceGradient:_gradient.content
                           sourceImage:sourceTensor.content
                         gradientState:_state
                   destinationGradient:destinationGradient.content];
    
    [self removeState];
    [self removeCachedImages];
    [self removeGradient];
    
    [sourceTensor.source setGradient:destinationGradient forwardTarget:self];
    [destinationGradient unlock];
    [sourceTensor.source gradientReadyFromForwardTarget:self onCommandBuffer:commandBuffer];
}

- (void)removeCachedGradients {
    [_inputGradients makeObjectsPerformSelector:@selector(unlock)];
    [_inputGradients removeAllObjects];
    
    DB_TRACE(-_verbose+3, "\n%s rm all gradients", self.labelUTF8);
}

- (void)removeGradient {
    DB_TRACE(-_verbose+2, "\n%s rm %s",
             self.labelUTF8,
             NSStringFromDataShape(_gradient.shape).UTF8String);
    
    [_gradient unlock];
    _gradient = nil;
}

- (void)reduceSumBatchGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    // Sum all of the gradients.
    int numOfGradients = self.numOfGradients;
    NSAssert(numOfGradients > 0, @"Invalid number of gradients.");
    
    if (numOfGradients == 1) {
        _gradient = _inputGradients[0];
        [_gradient lock];
        goto GRADIENT_SUM_FINISH;
    }
    else {
        MetalTensor temp = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape source:self commandBuffer:commandBuffer];
        if (_reduceSumOperation == nil) {
            _reduceSumOperation = [[MPSCNNAdd alloc] initWithDevice:_device];
        };
        MetalTensor t1 = _inputGradients[0];
        MetalTensor t2 = _inputGradients[1];
        _gradient = temp;
        [_reduceSumOperation encodeToCommandBuffer:commandBuffer primaryImage:t1.content secondaryImage:t2.content destinationImage:_gradient.content];
        if (numOfGradients == 2) {
            goto GRADIENT_SUM_FINISH;
        }
        
        for (int i = 2; i < numOfGradients; i++) {
            t1 = _inputGradients[i];
            t2 = _gradient;
            _gradient = _inputGradients[i-1];
            [_reduceSumOperation encodeToCommandBuffer:commandBuffer primaryImage:t1.content secondaryImage:t2.content destinationImage:_gradient.content];
        }
        [_gradient lock];
        [temp unlock];
        goto GRADIENT_SUM_FINISH;
    }
    
GRADIENT_SUM_FINISH:
    [self removeCachedGradients];
}

- (BOOL)isAllGradientsReceived {
    NSAssert(_targets.count >= _inputGradients.count, @"There are %d gradients expected, but got %d.", (int)_targets.count, (int)_inputGradients.count);
    return _inputGradients.count == _targets.count;
}

- (void)removeState {
    _state.readCount = 0;
    _state = nil;
}

#pragma mark - Forward Index

- (NSInteger)nextAvailableImageIndex {
    int bitCount = sizeof(unsigned long long)<<3;
    unsigned long long reservedForwardFlags = _reservedTargetFlags;
    int index = 0;
    while (reservedForwardFlags & 0x01ULL) {
        reservedForwardFlags = reservedForwardFlags >> 1;
        index ++;
        if (index >= bitCount) {
            NSAssert(NO, @"There is no place for new input image index.");
            return 0;
        }
    }
    return index;
}

- (void)reserveImageIndex:(NSInteger)index {
    NSParameterAssert(index >= 0 && index < _numOfImages);
    _reservedTargetFlags |= (0x01ULL<<index);
}

- (void)releaseImageIndex:(NSInteger)index {
    NSParameterAssert(index >= 0 && index < _numOfImages);
    _reservedTargetFlags &= (~((0x01ULL)<<index));
}

- (void)receiveImageAtIndex:(NSInteger)index {
    NSParameterAssert(index >= 0 && index < _numOfImages);
    _receivedImageFlags |= (0x01ULL<<index);
}

- (BOOL)isAllImagesReceived {
    return _receivedImageFlags == _reservedTargetFlags && _reservedTargetFlags > 0;
}

- (void)resetImagesReceivedFlags {
    _receivedImageFlags = 0x0ULL;
}

@end
