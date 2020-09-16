//
//  LandmarksNet.m
//  RapidFaceDetect
//
//  Created by Feng Stone on 2020/7/25.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import "LandmarksNet.h"
#import <MetalTensor/MetalTensorOutputLayer.h>

@interface LandmarksNet() {
    
    MetalTensorOutputLayer *_output;
    float32_t _result[136];
    float32_t _points[136];
    float32_t STD[136];
    float32_t MEAN[136];
}

@end

@implementation LandmarksNet

- (instancetype)init {
    NSString *infoPlist = [[NSBundle mainBundle] pathForResource:@"Landmarks" ofType:@"plist"];
    if (self = [super initWithPlist:infoPlist]) {
        NSString *std_mean_file = [[NSBundle mainBundle] pathForResource:@"std_mean" ofType:@"plist"];
        NSParameterAssert([[NSFileManager defaultManager] fileExistsAtPath:std_mean_file]);
        NSDictionary *std_mean_dict = [NSDictionary dictionaryWithContentsOfFile:std_mean_file];
//        NSLog(@"%@", std_mean_dict);
        NSString *std_string = [std_mean_dict[@"std"] stringByReplacingOccurrencesOfString:@" " withString:@""];
        NSArray<NSString *> *std_string_array = [std_string componentsSeparatedByString:@","];
        NSParameterAssert(std_string_array.count == 136);
        for (int i = 0; i < 136; i++) {
            STD[i] = std_string_array[i].floatValue;
        }
        
        NSString *mean_string = [std_mean_dict[@"mean"] stringByReplacingOccurrencesOfString:@" " withString:@""];
        NSArray<NSString *> *mean_string_array = [mean_string componentsSeparatedByString:@","];
        NSParameterAssert(mean_string_array.count == 136);
        for (int i = 0; i < 136; i++) {
            MEAN[i] = mean_string_array[i].floatValue;
        }
    }
    return self;
}

- (void)loadWeights {
    NSString *dataFile = [[NSBundle mainBundle] pathForResource:@"Landmarks" ofType:@"bin"];
    NSString *mapFile = [[NSBundle mainBundle] pathForResource:@"Landmarks" ofType:@"json"];
    [self loadWeights:dataFile mapFile:mapFile];
}

- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
    _output = (MetalTensorOutputLayer *)[self layerWithName:@"output"];
    NSParameterAssert(_output);
    
    __weak __auto_type weakSelf = self;
    self.scheduledHandler = ^(id<MTLCommandBuffer> _Nonnull cmd) {
//        [[FPSCounter sharedCounter] start];
    };
    self.completedHandler = ^(id<MTLCommandBuffer> _Nonnull cmd) {
            
        __strong __auto_type strongSelf = weakSelf;
        
//        [[FPSCounter sharedCounter] stop];
            
        DataShape *shape = [strongSelf->_output outputShapeRef];
        int size = ProductDepth4Divisible(shape);
        float16_t *result = malloc(size*sizeof(float16_t));
        MPSImage *outputImage = [strongSelf->_output outputImage];
        [outputImage toBuffer:(Byte *)result];
        ConvertFloat16To32(result, strongSelf->_result, size);
        free(result);
        
        [strongSelf processResult];
    };
}

- (void)processResult {
    
    const float32_t eps = 1e-6;
    for (int i = 0; i < 136; i ++) {
        float32_t v = _result[i] * STD[i] + MEAN[i];
        float32_t delta = fabsf(_points[i] - v)+eps;
        delta = delta / 2.0f;
        _points[i] = (_points[i] / delta + v * delta) / (delta + 1.0f/delta);
    }
    
    if ([_delegate respondsToSelector:@selector(LandmarksNet:didFinishWithPoints:)]) {
        [_delegate LandmarksNet:self didFinishWithPoints:_points];
    }
    
}

@end
