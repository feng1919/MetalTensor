//
//  MIConcatenateTarget.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/9.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MITemporaryImage.h"

NS_ASSUME_NONNULL_BEGIN

@class MetalTensorLayer;

@protocol MIConcatenateTarget <NSObject>

@required
- (MITemporaryImage *)concatenateTarget;
- (NSUInteger)channelOffsetForOutput:(MetalTensorLayer *)output;

@end

NS_ASSUME_NONNULL_END
