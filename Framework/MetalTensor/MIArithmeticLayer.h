//
//  MIArithmeticLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/3.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@class MIMPSImage;
@interface MIArithmeticLayer : MetalTensorLayer {
    
@public
    MPSCNNArithmetic *arithmetic;
    DataShape dataShape;
}
@property (nonatomic, strong, nullable) MIMPSImage *secondaryImage;

+ (instancetype)additionArithmeticLayerWithDataShape:(DataShape *)dataShape;
+ (instancetype)subtractionArithmeticLayerWithDataShape:(DataShape *)dataShape;
+ (instancetype)multiplicationArithmeticLayerWithDataShape:(DataShape *)dataShape;
+ (instancetype)divisionArithmeticLayerWithDataShape:(DataShape *)dataShape;

- (void)setDestinationFeatureChannelOffset:(NSInteger)channelOffset;

@end

NS_ASSUME_NONNULL_END
