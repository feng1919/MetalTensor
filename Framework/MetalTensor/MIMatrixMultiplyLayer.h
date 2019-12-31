//
//  MIMatrixMultiplyLayer.h
//  MetalTensor
//
//  Created by Feng Stone on 2019/12/31.
//  Copyright © 2019 fengshi. All rights reserved.
//

/**
 *  To make a matrix multiplication between two MPS tensors.
 *  The multiplication is virtually procuced by multi depth-wise convolutions.
 *  The two tensors have the same shape, the number of channels is 'n'.
 *  First, tile each channel of the right tensor by n times.
 *  And then make the depth-wise convolution n times between left tensor and
 *  n tiled right tensors.
 *  Finally concatenate the results.
 *
 *  We use this layer to make Gram Matrix, which requires tensors channels inner
 *  product to represent the features of current convolution, so we may continue
 *  to calculate on GPU within the convolution neural networks without copy the
 *  data to CPU space to make logic computation, which is expensive and chewing
 *  up the performace.
 *
 */

#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIMatrixMultiplyLayer : MetalTensorLayer

@end

NS_ASSUME_NONNULL_END
