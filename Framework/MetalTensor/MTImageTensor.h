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

- (instancetype)initWithImage:(UIImage *)image normalized:(BOOL)normalized;

@end

NS_ASSUME_NONNULL_END
