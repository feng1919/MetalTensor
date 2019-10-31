//
//  MetalTensorLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/7/4.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"
#import "MITemporaryImageCache.h"
#include "numpy.h"
#import "MPSImage+Extension.h"

@implementation MetalTensorLayer

- (instancetype)init {
    NSAssert(NO, @"Invalid initialize method.");
    return nil;
}

- (instancetype)initWithInputShapes1:(DataShape *_Nonnull*_Nonnull)inputShapes size:(int)size {
    
    if (self = [super init]) {
        
        NSAssert(size > 0, @"Invalid inputs number, if layer don't have any input, use MetalTensorNode.");
        
        _inputShapes = malloc(size * sizeof(DataShape));
        for (int i = 0; i < size; i++) {
            _inputShapes[i] = inputShapes[i][0];
        }
        
        _numOfInputs = size;
        
        _reservedFlags = calloc(size, sizeof(BOOL));
        _receivedFlags = calloc(size, sizeof(BOOL));
        _inputs = [NSMutableDictionary dictionaryWithCapacity:size];
        
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
        
        _inputShapes = malloc(size * sizeof(DataShape));
        npmemcpy(_inputShapes, inputShapes, size*sizeof(DataShape));
        
        _numOfInputs = size;
        
        _reservedFlags = calloc(size, sizeof(BOOL));
        _receivedFlags = calloc(size, sizeof(BOOL));
        _inputs = [NSMutableDictionary dictionaryWithCapacity:size];
        
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
    
    free(_receivedFlags);
    _receivedFlags = NULL;
    
    free(_reservedFlags);
    _reservedFlags = NULL;
}

#pragma mark - GET

- (DataShape *)inputShapes {
    return _inputShapes;
}

- (int)numOfInputs {
    return _numOfInputs;
}

#pragma mark - Management of input indices

- (BOOL)isAllReceived {
    for (int i = 0; i < _numOfInputs; i++) {
        if (_receivedFlags[i] == NO) {
            return NO;
        }
    }
    return YES;
}

- (void)resetReceivedFlags {
    for (int i = 0; i < _numOfInputs; i++) {
        _receivedFlags[i] = NO;
    }
}

- (NSInteger)nextAvailableTextureIndex {
    for (int i = 0; i < _numOfInputs; i++) {
        if (_reservedFlags[i] == NO) {
            return i;
        }
    }
    NSAssert(NO, @"There is no place for new input index.");
    return 0;
}

- (void)reserveTextureIndex:(NSInteger)index {
    _reservedFlags[index] = YES;
}

- (void)releaseTextureIndex:(NSInteger)index {
    _reservedFlags[index] = NO;
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    NSParameterAssert(_numOfInputs > 0);
    _outputShape = _inputShapes[0];
}

#pragma mark - MetalLayerInput delegate

- (void)setInputImage:(MITemporaryImage *)newInputImage atIndex:(NSInteger)imageIndex {
    NSAssert(imageIndex < _numOfInputs, @"Invalid input image index.");
    NSAssert(DataShapesTheSame(&_inputShapes[imageIndex], [newInputImage shape]), @"The input image's shape is not identical to the inputShapes[%d]", (int)imageIndex);
    _inputs[@(imageIndex)] = newInputImage;
    [newInputImage lock];
    
    DB_TRACE(-_verbose+2, "\n%s(%ld) <-- %s", self.labelUTF8, imageIndex, NSStringFromDataShape(newInputImage.shape).UTF8String);
}

- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)cmdBuf {
    if ([self isAllReceived]) {
        return;
    }
    
    _receivedFlags[imageIndex] = YES;
    
    if ([self isAllReceived]) {
        [self processTensorWithCommandBuffer:cmdBuf];
        [self resetReceivedFlags];
    }
}

- (void)processTensorWithCommandBuffer:(id<MTLCommandBuffer>)cmdBuf {
    
    NSAssert(NO, @"It should be overrided by sub classes.");
    
    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:cmdBuf];
    [_outputTempImage newTemporaryImageForCommandBuffer:cmdBuf];
    
//    for (int i = 0; i < _numOfInputs; i++) {
//        MITemporaryImage *tensor = _images[i];
//        [neuron setDestinationFeatureChannelOffset:offsets[i]];
//        [neuron encodeToCommandBuffer:cmdBuf sourceImage:tensor.image destinationImage:_outputTempImage.image];
//        [tensor unlock];
//    }
    
    [self removeCachedImages];
    
    [self notifyTargetsAboutNewTempImage:cmdBuf];
}

- (void)removeCachedImages {
    [_inputs.allValues makeObjectsPerformSelector:@selector(unlock)];
    [_inputs removeAllObjects];
    
    DB_TRACE(-_verbose+3, "\n%s rm all inputs", self.labelUTF8);
}

@end

MetalTensorLayer *ConnectLinearLayers(NSArray<MetalTensorLayer *> *layers) {
    if (layers.count <= 1) {
        return layers.firstObject;
    }
    MetalTensorLayer *latest = layers.firstObject;
    for (int i = 1; i < layers.count; i++) {
        MetalTensorLayer *newLayer = layers[i];
        [latest addTarget:newLayer];
        latest = newLayer;
    }
    return latest;
}
