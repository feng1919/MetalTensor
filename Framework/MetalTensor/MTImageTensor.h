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
    
@public
#if DEBUG
    float16_t *_result;
#endif
    
@protected
    MPSImage *_mpsImage;
}

@property (nonatomic, strong) MPSImage *mpsImage;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithShape:(DataShape *)shape NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithShape:(DataShape *)shape dataType:(MPSDataType)dataType NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithContent:(MPSImage *)mpsImage NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithImage:(UIImage *)image normalized:(BOOL)normalized;
- (instancetype)initWithImage:(UIImage *)image normalized:(BOOL)normalized frameBuffer:(BOOL)frameBuffer;
- (instancetype)initWithImage:(UIImage *)image normalized:(BOOL)normalized frameBuffer:(BOOL)frameBuffer rgb:(BOOL)rgb NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithShape:(DataShape *)shape dataFormat:(MPSDataType)dataFormat numberOfImage:(NSUInteger)numberOfImages NS_UNAVAILABLE;

- (void)loadData:(float16_t *)data length:(NSInteger)length;
- (float16_t *)buffer;
- (void)loadBuffer;

#if DEBUG

- (void)dump;
- (void)printResult;
- (void)printResultCHW;
- (void)printResultHWC;
- (void)innerChannelsProduct;
- (void)printPixelAtX:(int)x Y:(int)y;
- (void)printPixelsFromX:(int)x0 toX:(int)x1 fromY:(int)y0 toY:(int)y1;
- (void)checkNan;
- (void)checkInf;

#endif

@end

NS_ASSUME_NONNULL_END
