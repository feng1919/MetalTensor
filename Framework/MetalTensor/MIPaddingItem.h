//
//  MIPaddingItem.h
//  MetalTensor
//
//  Created by Feng Stone on 2019/11/6.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "metal_tensor_structures.h"

NS_ASSUME_NONNULL_BEGIN

@class MIPaddingItem;
extern MIPaddingItem *MPSPaddingTensorFlowSame;
extern MIPaddingItem *MPSPaddingValid;
extern MIPaddingItem *MPSPaddingFull;

@interface MIPaddingItem : NSObject <MPSNNPadding>

@property (nonatomic, assign, readonly) MTPaddingMode padding;

- (instancetype)initWithPaddingMode:(MTPaddingMode)mode;

MIPaddingItem *SharedPaddingItemWithMode(MTPaddingMode mode);

@end

NS_ASSUME_NONNULL_END
