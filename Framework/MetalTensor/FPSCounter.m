//
//  FPSCounter.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/27.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "FPSCounter.h"
#import <QuartzCore/QuartzCore.h>

static FPSCounter *_fpsCounter = nil;

@interface FPSCounter () {
    
    int *_fps_list;
    int _average_count;
    int _index;
    CFTimeInterval _start_time_stamp;
}

@end

@implementation FPSCounter

+ (void)initialize {
    if (self == [FPSCounter class]) {
        _fpsCounter = [[FPSCounter alloc] init];
    }
}

- (instancetype)init {
    if (self = [super init]) {
        _average_count = 16;
        _index = 0;
        _fps_list = calloc(_average_count, sizeof(int));
    }
    return self;
}

- (void)dealloc {
    free(_fps_list);
}

+ (id)sharedCounter {
    return _fpsCounter;
}

- (void)setAverageCount:(int)count {
    NSParameterAssert(count > 0);
    
    @synchronized (self) {
        if (_fps_list) {
            free(_fps_list);
        }
        _average_count = count;
        _index = 0;
        _start_time_stamp = 0.0f;
        _fps_list = calloc(_average_count, sizeof(int));
    }
}

- (void)start {
    _start_time_stamp = CACurrentMediaTime();
}

- (void)stop {
    @synchronized (self) {
        CFTimeInterval interval = CACurrentMediaTime()-_start_time_stamp;
        int fps = ceilf(1.0f/interval);
        _index ++;
        if (_index == _average_count) {
            _index = 0;
        }
        
        _fps_list[_index] = fps;
    }
}

- (void)reset {
    @synchronized (self) {
        if (_fps_list) {
            free(_fps_list);
        }
        _fps_list = calloc(_average_count, sizeof(int));
        _start_time_stamp = 0.0f;
        _index = 0;
    }
}

- (int)FPS {
    @synchronized (self) {
        int total_fps = 0;
        for (int i = 0; i < _average_count; i++) {
            total_fps += _fps_list[i];
        }
        return total_fps/_average_count;
    }
}

- (void)printFPS {
    int fps = [self FPS];
    NSLog(@"FPS: %d", fps);
}

@end
