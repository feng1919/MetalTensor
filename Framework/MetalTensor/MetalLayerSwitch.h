//
//  MetalLayerSwitch.h
//  MetalImage
//
//  Created by Feng Stone on 2019/8/1.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorNode.h"

NS_ASSUME_NONNULL_BEGIN

/*
 *  To decide which head to be actived.
 */

@interface MetalLayerSwitch : MetalTensorNode <MetalTensorInput>

@property (atomic, assign) unsigned int activedTargetIndex;

@end

NS_ASSUME_NONNULL_END
