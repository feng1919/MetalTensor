//
//  MIGramMatrixLayer.h
//  MetalTensor
//
//  Created by Feng Stone on 2019/12/31.
//  Copyright © 2019 fengshi. All rights reserved.
//

/*
 *  This layer make a matrix multiplication between channels
 *  aka Gram Matrix.
 *
 */

#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIGramMatrixLayer : MetalTensorLayer

/*
 *  The output result could be big values over 10^8,
 *  and the maximum value of float16_t is 65530, this
 *  parameter is used for scale down the scalar.
 *  Set it before -compile: get called, it will not
 *  take effect after compiled.
 *  Default by 1.0f.
 */
@property (nonatomic, assign) float weight;

@end

NS_ASSUME_NONNULL_END
