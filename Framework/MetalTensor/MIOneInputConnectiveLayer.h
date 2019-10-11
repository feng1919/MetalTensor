//
//  MIOneInputConnectiveLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIOneInputConnectiveLayer : MetalTensorLayer <MILayerInput> {
    
@protected
    DataShape _firstInputShape;
    MITemporaryImage *firstInputImage;
}

@property (atomic, assign) DataShape firstInputShape;

- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape;
- (void)processTensorWithCommandBuffer:(id<MTLCommandBuffer>)cmdBuf;

// for convenience
// return the last layer
MIOneInputConnectiveLayer *ConnectLinearLayers(NSArray<MIOneInputConnectiveLayer *> *layers);

@end

NS_ASSUME_NONNULL_END
