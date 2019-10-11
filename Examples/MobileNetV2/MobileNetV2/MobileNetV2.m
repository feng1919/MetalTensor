//
//  MobileNetV2.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/22.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MobileNetV2.h"
#import "ImageNet1000.h"
#import <UIKit/UIKit.h>
#import <MetalTensor/MetalTensor.h>

@interface MobileNetV2() {
    
    MetalTensorOutputLayer *_output;
    DataShape _outputShape;
    int _outputSize;
}

@end

@implementation MobileNetV2

- (instancetype)init {
    NSString *infoPlist = [[NSBundle mainBundle] pathForResource:@"MobileNetV2_1.0" ofType:@"plist"];
    return [self initWithPlist:infoPlist];
}

- (void)loadWeights {
    NSString *dataFile = [[NSBundle mainBundle] pathForResource:@"MobileNetV2_1.0" ofType:@"bin"];
    NSString *mapFile = [[NSBundle mainBundle] pathForResource:@"MobileNetV2_1.0" ofType:@"json"];
    [self loadWeights:dataFile mapFile:mapFile];
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    _output = _outputLayers.firstObject;
    NSParameterAssert(_output);
    _outputShape = _output.outputShape;
    _outputSize = _outputShape.row * _outputShape.column * _outputShape.depth;
    
    __weak __auto_type weakSelf = self;
    self.scheduledHandler = ^(id<MTLCommandBuffer> cmd) {
        [[FPSCounter sharedCounter] start];
    };
    self.completedHandler = ^(id<MTLCommandBuffer> cmd) {
        
        [[FPSCounter sharedCounter] stop];
        
        __strong __auto_type strongSelf = weakSelf;
        int size = strongSelf->_outputSize;
        float16_t *result = malloc(size*sizeof(float16_t));
        MPSImage *outputImage = [strongSelf->_output outputImage];
        [outputImage toFloat16Array:result];
        ConvertFloat16To32(result, [[ImageNet1000 sharedInstance] rateBuffer], size);
        free(result);
        
        NSDictionary *rank5 = [[ImageNet1000 sharedInstance] rank5];
        [[NSNotificationCenter defaultCenter] postNotificationName:MOBILENET_PREDICTING_RESULT object:rank5];
    };
}

@end

NSNotificationName MOBILENET_PREDICTING_RESULT = @"NSNotificationName MOBILENET_PREDICTING_RESULT =";
