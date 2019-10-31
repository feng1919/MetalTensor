//
//  PortraitSegmentNet.m
//  MetalTensorDemo
//
//  Created by Feng Stone on 2019/9/25.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "PortraitSegmentNet.h"
#import <MetalImage/UIImage+Texture.h>
#import <MetalImage/MetalDevice.h>
#import <MetalTensor/FPSCounter.h>

@interface PortraitSegmentNet()
{
    MetalTensorOutputLayer *_output;
    DataShape _outputShape;
    float16_t *_outputBuffer;
}

@end

@implementation PortraitSegmentNet

- (instancetype)init {
    NSString *infoPlist = [[NSBundle mainBundle] pathForResource:@"PortraitSegmentNet.0.5" ofType:@"plist"];
    self = [self initWithPlist:infoPlist];
    return self;
}

- (void)dealloc {
    if (_outputBuffer) {
        free(_outputBuffer);
    }
}

- (void)loadWeights {
    NSString *dataFile = [[NSBundle mainBundle] pathForResource:@"PortraitSegmentNet.0.5" ofType:@"bin"];
    NSString *mapFile = [[NSBundle mainBundle] pathForResource:@"PortraitSegmentNet.0.5" ofType:@"json"];
    [self loadWeights:dataFile mapFile:mapFile];
}

- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
#if DEBUG
    self.verbose = 0;
#endif
    
    _output = _outputLayers.firstObject;
    NSParameterAssert(_output);
    _outputShape = _output.outputShape;
    
    int size = ProductOfDataShape(&_outputShape);
    NSParameterAssert(size > 0);
    
    if (_outputBuffer) {
        free(_outputBuffer);
    }
    _outputBuffer = malloc(size * sizeof(float16_t));
    
    __weak __auto_type weakSelf = self;
    self.scheduledHandler = ^(id<MTLCommandBuffer> cmd) {
        [[FPSCounter sharedCounter] start];
    };
    self.completedHandler = ^(id<MTLCommandBuffer> cmd) {
        [[FPSCounter sharedCounter] stop];
        
        [[NSNotificationCenter defaultCenter] postNotificationName:PortraitSegmentNetDidFinish object:nil];
        
        __strong __auto_type strongSelf = weakSelf;
        
        MPSImage *outputImage = [strongSelf->_output outputImage];
        [outputImage toFloat16Array:strongSelf->_outputBuffer];
        clamp_mask(strongSelf->_outputBuffer, size);
        [strongSelf->_delegate PortraitSegmentNet:strongSelf
                                    predictResult:strongSelf->_outputBuffer];
    };
}

- (MTLUInt2)outputMaskSize {
    return MTLUInt2Make(_outputShape.column, _outputShape.row);
}

void clamp_mask(float16_t *mask, const unsigned int size)
{
    const float16_t threshold = 0.1f;
    const float16_t epsilon = 1e-4;
    float16_t std = 1.0f-threshold*2.0f;
    
    for (int i = 0; i < size; i++) {
        mask[i] = fminf(fmaxf(mask[i], threshold), 1.0f-threshold);
        mask[i] -= threshold;
        mask[i] /= std;
        
        mask[i] = fminf(fmaxf(mask[i], epsilon), 1.0f-epsilon);
    }
}

@end

NSNotificationName PortraitSegmentNetDidFinish = @"NSNotificationName PortraitSegmentNetDidFinish = ";
