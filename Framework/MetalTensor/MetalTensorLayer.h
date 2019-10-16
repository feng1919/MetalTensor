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
     *  input tensors and shapes
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

/*
 *  Recommanded initialze functions of MetalTensorLayer.
 *  The method -init is invalid, for MetalTensorLayer
 *  there always be an input at least.
 */
- (instancetype)initWithInputShape:(DataShape *)inputShape;
- (instancetype)initWithInputShapes:(DataShape *_Nonnull)inputShapes size:(int)size;
- (instancetype)initWithInputShapes1:(DataShape *_Nonnull*_Nonnull)inputShapes size:(int)size;

/*
 *  The input shapes for the MetalTensorLayer instance.
 */
- (DataShape * _Nonnull)inputShapes;

/*
 *  Number Of inputs for the MetalTensorLayer instance.
 */
- (int)numOfInputs;

/*
 *  Initialze assignment to variables.
 */
- (void)initialize;

/*
 *  Remove all cached tensors after the calculation.
 */
- (void)removeCachedImages;

@end

// for convenience
// return the last layer
MetalTensorLayer *ConnectLinearLayers(NSArray<MetalTensorLayer *> *layers);

NS_ASSUME_NONNULL_END
