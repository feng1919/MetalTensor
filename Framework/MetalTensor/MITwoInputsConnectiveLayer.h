//
//  MITwoInputsConnectiveLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/3.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIOneInputConnectiveLayer.h"

NS_ASSUME_NONNULL_BEGIN

typedef struct {
    BOOL targetHasBeenSet;
    BOOL hasReceivedFrame;
}DataFlowFlags;

@interface MITwoInputsConnectiveLayer : MIOneInputConnectiveLayer {
    
@protected
    DataShape _secondInputShape;
    MITemporaryImage *secondInputImage;
    DataFlowFlags _firstDataFlags;
    DataFlowFlags _secondDataFlags;
}

@property (atomic, assign) DataShape secondInputShape;

- (instancetype)initWithFirstInputShape:(DataShape *)firstInputShape
                       secondInputShape:(DataShape *)secondInputShape
                            outputShape:(DataShape *)outputShape;

@end

NS_ASSUME_NONNULL_END
