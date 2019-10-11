//
//  MobileNetV2.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/22.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalTensor/MetalNeuralNetwork.h>

NS_ASSUME_NONNULL_BEGIN

@interface MobileNetV2 : MetalNeuralNetwork

@end

extern NSNotificationName MOBILENET_PREDICTING_RESULT;

NS_ASSUME_NONNULL_END
