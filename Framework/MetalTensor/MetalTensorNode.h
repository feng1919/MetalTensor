//
//  MetalTensorNode.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <MetalImage/MetalImageFunction.h>
#import "MetalTensorProtocols.h"

NS_ASSUME_NONNULL_BEGIN

/*
 *  A MetalTensorNode makes the connection to the other nodes of the
 *  network, if this node should pass a tensor to the nodes connected
 *  with, or receive tensors from those, then it should confirm to
 *  <MTForwardDelegate> protocol for forward propagation and to
 *  <MTBackwardDelegate> protocol for backward propagation.
 */

@interface MetalTensorNode : NSObject {
    
@protected
    
    //  Targets for forward propagation.
    NSMutableArray <ForwardTarget> *_targets;
    //  The index of slot of the tensor of the forward target node
    //  received.
    NSMutableArray <NSNumber *> *_targetIndices;
    
    //  A string to identify the node.
    NSString *_label;
    
    //  The device to create the node on.
    id<MTLDevice> _device;
    
    //  Flag of backward propagation will be cast by the node.
    BOOL _needBackward;
    
//#ifdef DEBUG
    // If 0, there will be no 'printf' log in console.
    int _verbose;
//#endif
}

/*
 *  A string to identify the node.
 */
@property (nonatomic, strong, nullable) NSString *label;

/*
 *  Whether this layer need backward propagation.
 *  If YES, there will be extra computation of tensor gradients.
 *  This framework is for inference purpose, so there will be no
 *  computation of weights, bias or batch normalization parameters,
 *  etc.
 *  The default is NO.
 */
@property (nonatomic, assign) BOOL needBackward;

/*
 *  One should override this method and build up the node, such
 *  as initialize the kernel operation, configue the data shape of
 *  tensor for output and offsets of convolution, etc.
 */
- (void)compile:(id<MTLDevice>)device NS_REQUIRES_SUPER;

//  FORWARD CONNECTING...
- (NSArray<ForwardTarget> *)targets;
- (void)addTarget:(ForwardTarget)newTarget;
- (void)addTarget:(ForwardTarget)newTarget atIndex:(NSInteger)imageIndex;
- (void)removeTarget:(ForwardTarget)targetToRemove;
- (void)removeAllTargets;

//#ifdef DEBUG
@property (nonatomic, assign) int verbose;
- (const char*)labelUTF8;
//#endif

@end

NS_ASSUME_NONNULL_END
