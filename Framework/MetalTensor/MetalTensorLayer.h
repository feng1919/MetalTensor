//
//  MetalTensorLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/7/4.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorNode.h"
#import "MTTensor.h"
#import "MTImageTensor.h"

NS_ASSUME_NONNULL_BEGIN

@interface MetalTensorLayer : MetalTensorNode <MTForwardDelegate, MTBackwardDelegate> {
    
@protected
    
    //  The data shape of forward result tensor.
    DataShape _outputShape;
    
    //  input image shapes
    DataShape *_inputShapes;
    //  number of input images
    int _numOfImages;
    //  input images
    NSMutableDictionary<NSNumber *, MetalTensor> *_inputImages;
    //  Forward result tensor.
    MetalTensor _image;
    //  Forward operation.
    MPSCNNKernel *_operation;
    
    //  The number of input gradients is the same as the number of
    //  forward targets.
    //  input gradients
    NSMutableArray<MetalTensor> *_inputGradients;
    //  Backward gradient tensor, the gradient used to compute
    //  new gradients.
    MetalTensor _gradient;
    //  The forward operation MPS state.
    MPSState *_state;
    //  The -encode operation for gradient calculation.
    MPSCNNGradientKernel *_gradientOp;
    
    //  The flags to manage input tensors.
    //  Bitmask
    unsigned long long _receivedImageFlags;
    
    //  The flags to manage the forward target indices.
    //  Bitmask
    unsigned long long _reservedTargetFlags;
}

/*
 *  The output shape to forward target.
 *  One may specified an output shape in plist file, MetalTensor
 *  framework will generate one if it is not specified.
 *  One may modify this after compiling.
 *
 */
@property (nonatomic, assign) DataShape outputShape;

/*
 *  If YES, the node will not back propagate the gradients, but
 *  output the gradients.
 *  Default by NO.
 */
@property (nonatomic, assign) BOOL stopGradient;

/*
 *  If stopGradient is set to YES, the processing result of gradients
 *  computation will be saved to _savedGradients.
 */
@property (nonatomic, strong, nullable) MTImageTensor *savedGradients;

/*
 *  Unavailable.
 */
- (instancetype)init NS_UNAVAILABLE;

/*
 *  Recommanded initialze functions of MetalTensorLayer.
 *  The method -init is invalid, for MetalTensorLayer
 *  there always be one input at least.
 */
- (instancetype)initWithInputShapes:(DataShape *)inputShapes size:(int)size NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithInputShapes1:(DataShape *_Nonnull*_Nonnull)inputShapes size:(int)size NS_DESIGNATED_INITIALIZER;

/*
 *  One input initialize.
 */
- (instancetype)initWithInputShape:(DataShape *)inputShape;

/*
 *  Return an empty neuron.
 *  You may use this for blitting the tensors at run time.
 */
- (MPSCNNNeuron *)blit;

/*
 *  The input shapes for the MetalTensorLayer instance.
 */
- (DataShape *)inputShapes;

/*
 *  Calculate output data shape.
 */
- (void)updateOutputShape;

/*
 *  Number Of input images for the MetalTensorLayer instance.
 */
- (int)numOfImages;

/*
 *  Number Of input gradients for the MetalTensorLayer instance.
 *  The same as the number of forward targets.
 */
- (int)numOfGradients;

/*
 *  Subclass override this method to initialize assignments to variables.
 */
- (void)initialize;

/*
 *  Remove all cached tensors after the calculation.
 */
- (void)removeCachedImages;

/*
 *  Remove all cached gradients after the reduce sum .
 */
- (void)removeCachedGradients;

///////////////////////////////////////////////////////////////////////
//  FORWARD
- (void)removeImage;
- (void)setImageToTargets;
- (void)notifyTargetsAboutNewImageOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;

///////////////////////////////////////////////////////////////////////
//  BACKWARD
- (void)removeGradient;
- (void)removeState;
- (void)reduceSumBatchGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;

#if DEBUG
- (void)printInputAndOutputShapes;

@property (nonatomic, assign) BOOL dumpResult;
@property (nonatomic, strong, nullable) MTImageTensor *savedResult;
- (void)saveTensor:(MetalTensor)tensor onCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;

#endif

@end

NS_ASSUME_NONNULL_END
