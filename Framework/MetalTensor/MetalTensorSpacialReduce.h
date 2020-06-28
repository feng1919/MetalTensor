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

@interface MetalTensorSpacialReduce : NSObject

@property (nonatomic, assign) MPSDataType dataType;
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
