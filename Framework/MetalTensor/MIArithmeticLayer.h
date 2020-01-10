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

@property (nonatomic, strong) MPSCNNArithmetic *arithmetic;

@property (nonatomic, assign) float bias;
@property (nonatomic, assign) float primaryScale;
@property (nonatomic, assign) float secondaryScale;
@property (nonatomic, assign) MTLInt3 primaryStrides;
@property (nonatomic, assign) MTLInt3 secondaryStrides;
@property (nonatomic, assign) NSInteger channelOffset;

+ (instancetype)arithmeticLayerWithDataShape:(DataShape *)dataShape;
- (void)updateArithmeticParameters;
+ (Class)arithmeticWithType:(NSString *)arithmetic;

@end

NS_ASSUME_NONNULL_END
