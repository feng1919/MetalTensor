//
//  FPSCounter.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/27.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface FPSCounter : NSObject

+ (FPSCounter *)sharedCounter;

- (void)setAverageCount:(int)count;

- (void)start;
- (void)stop;
- (void)reset;
- (int)FPS;
- (void)printFPS;

@end

NS_ASSUME_NONNULL_END
