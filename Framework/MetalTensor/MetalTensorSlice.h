//
//  MetalTensorSlice.h
//  MetalTensor
//
//  Created by Feng Stone on 2020/1/11.
//  Copyright © 2020 fengshi. All rights reserved.
//

/*
 *  Extract one channel at givin offset and tile it to fill
 *  entire space.
 */
#import "MPSImage+Extension.h"
#import "metal_tensor_structures.h"
#import "MTTensor.h"

NS_ASSUME_NONNULL_BEGIN

@interface MetalTensorSlice : NSObject

@property (nonatomic, assign) MPSDataType dataType;

- (instancetype)initWithNumberOfChannel:(int)numberOfChannel;
- (void)compile:(id<MTLDevice>)device;
- (MetalTensor)sliceTensor:(MetalTensor)tensor channelIndex:(int)channelIndex commandBuffer:(id<MTLCommandBuffer>)commandBuffer;
- (void)sliceTensor:(MetalTensor)src toTensor:(MetalTensor)dst channelIndex:(int)channelIndex commandBuffer:(id<MTLCommandBuffer>)commandBuffer;

@end

NS_ASSUME_NONNULL_END
