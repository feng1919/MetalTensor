//
//  MetalTensorSpacialReduce.h
//  MetalTensor
//
//  Created by Feng Stone on 2020/1/13.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import "MPSImage+Extension.h"
#import "metal_tensor_structures.h"
#import "MTTensor.h"
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(unsigned, ReduceAxisMask) {
    ReduceAxisRow = 0x01U,
    ReduceAxisColumn = 0x01U<<1,
};

typedef NS_ENUM(int, ReduceType) {
    ReduceTypeMean = 0,
    ReduceTypeSum,
};

@interface MetalTensorSpacialReduce : NSObject

@property (nonatomic, assign) ReduceAxisMask axis;
@property (nonatomic, assign) ReduceType type;
@property (nonatomic, assign) DataShape inputShape;
@property (nonatomic, assign) MTLRegion clipRect;

- (void)compile:(id<MTLDevice>)device;
- (void)reduceOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                 sourceTensor:(MetalTensor)sourceTensor
            destinationTensor:(MetalTensor)destinationTensor;

@end

NS_ASSUME_NONNULL_END
