//
//  MIHistogramGenerator.h
//  MetalImage
//
//  Created by fengshi on 2017/9/8.
//  Copyright © 2017年 fengshi. All rights reserved.
//

#import "MetalImageFilter.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIHistogramGenerator : MetalImageFilter

@property (nonatomic, assign) MTLFloat4 backgroundColor;

@end

NS_ASSUME_NONNULL_END
