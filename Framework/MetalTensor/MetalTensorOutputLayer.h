//
//  MetalTensorOutputLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/5.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MetalTensorOutputLayer : MetalTensorLayer

@property (nonatomic, strong) MPSImage *outputImage;

- (instancetype)initWithOutputShape:(DataShape *)outputShape;

@end

NS_ASSUME_NONNULL_END
