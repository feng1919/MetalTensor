//
//  MTImageTensor.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/25.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MTTensor.h"
#import <UIKit/UIKit.h>

/*
 *  Load tensor from UIImage object.
 *
 */

NS_ASSUME_NONNULL_BEGIN

@interface MTImageTensor : MTTensor {
    
@protected
    MPSImage *_mpsImage;
}

@property (nonatomic, strong) MPSImage *mpsImage;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithShape:(DataShape *)image NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithContent:(MPSImage *)mpsImage NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithImage:(UIImage *)image normalized:(BOOL)normalized;
- (instancetype)initWithImage:(UIImage *)image normalized:(BOOL)normalized frameBuffer:(BOOL)frameBuffer NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithShape:(DataShape *)shape dataFormat:(TensorDataFormat)dataFormat numberOfImage:(NSUInteger)numberOfImages NS_UNAVAILABLE;

- (void)loadData:(float16_t *)data length:(NSInteger)length;
- (float16_t *)buffer;
- (void)loadBuffer;

#if DEBUG

- (void)dump;
- (void)printResult;
- (void)printPixelAtX:(int)x Y:(int)y;
- (void)printPixelsFromX:(int)x0 toX:(int)x1 fromY:(int)y0 toY:(int)y1;

#endif

@end

NS_ASSUME_NONNULL_END
