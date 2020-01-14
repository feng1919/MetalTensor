//
//  MIReduceUnaryLayer.h
//  MetalTensor
//
//  Created by Feng Stone on 2019/12/31.
//  Copyright © 2019 fengshi. All rights reserved.
//

/*
 *  To cast an unary reduce operation on the tensor.
 *  Supported operation: 'Max', 'Min', 'Mean' and 'Sum'.
 *  Supported axis: 'Row', 'Column' and 'Depth'.
 *
 */

#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIReduceUnaryLayer : MetalTensorLayer

@property (nonatomic, assign) ReduceType type;
@property (nonatomic, assign) ReduceAxisMask axis;

@end

NS_ASSUME_NONNULL_END
