//
//  MIArithmeticLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/3.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@class MTImageTensor;
@interface MIArithmeticLayer : MetalTensorLayer

@property (nonatomic, strong, nullable) MTImageTensor *secondaryImage;
@property (nonatomic, strong) NSString *arithmeticType;
@property (nonatomic, assign) NSInteger channelOffset;

@property (nonatomic, strong) MPSCNNArithmetic *arithmetic;

+ (instancetype)arithmeticLayerWithDataShape:(DataShape *)dataShape;

+ (Class)arithmeticWithType:(NSString *)arithmetic;

@end

NS_ASSUME_NONNULL_END
