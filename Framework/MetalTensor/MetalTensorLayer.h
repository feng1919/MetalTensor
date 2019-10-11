//
//  MetalTensorLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/7/4.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorNode.h"

NS_ASSUME_NONNULL_BEGIN

@interface MetalTensorLayer : MetalTensorNode <MetalTensorInput> {
    
@protected
    /*
     *  input information
     *  tensors and shapes
     */
    DataShape *_inputShapes;
    int _numOfInputs;
    NSMutableDictionary<NSNumber *, MITemporaryImage *> *_inputs;
    
    /*
     *  the flags used to manage the target indices.
     */
    BOOL *_reservedFlags;
    BOOL *_receivedFlags;
}

@property (nonatomic, assign, readonly) DataShape *inputShapes;
@property (nonatomic, assign, readonly) int numOfInputs;

- (instancetype)initWithInputShape:(DataShape *)inputShape outputShape:(DataShape *)outputShape;
- (instancetype)initWithInputShapes:(DataShape *_Nonnull)inputShapes size:(int)size outputShape:(DataShape *)outputShape;
- (instancetype)initWithInputShapes1:(DataShape *_Nonnull*_Nonnull)inputShapes size:(int)size outputShape:(DataShape *)outputShape;

- (void)removeCachedImages;

@end

// for convenience
// return the last layer
MetalTensorLayer *ConnectLinearLayers(NSArray<MetalTensorLayer *> *layers);

NS_ASSUME_NONNULL_END
