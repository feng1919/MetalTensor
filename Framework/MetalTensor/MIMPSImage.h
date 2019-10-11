//
//  MIMPSImage.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/25.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MITemporaryImage.h"
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface MIMPSImage : MITemporaryImage {
    
@protected
    MPSImage *_mpsImage;
}

@property (nonatomic, strong) MPSImage *mpsImage;

- (instancetype)initWithImage:(UIImage *)image normalized:(BOOL)normalized;

@end

NS_ASSUME_NONNULL_END
