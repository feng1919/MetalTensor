//
//  MetalTensorNode.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <MetalImage/MetalImageFunction.h>
#import "MetalTensorInput.h"
#import "MITemporaryImage.h"

NS_ASSUME_NONNULL_BEGIN

/*
 *  Connecting all of the network operations, layers and tensors.
 */

@interface MetalTensorNode : NSObject {
    
@protected
    DataShape _outputShape;
    MITemporaryImage *_outputTempImage;
    NSMutableArray <id<MetalTensorInput>> *_targets;
    NSMutableArray <NSNumber *> *_targetTempImageIndices;
    
    NSString *_label;
    
    id<MTLDevice> _device;
    
//#ifdef DEBUG
    // If 0, there will be no 'printf' log in console.
    int _verbose;
//#endif
}

@property (nonatomic, strong, nullable) NSString *label;
@property (nonatomic, readonly) DataShape outputShape;
@property (nonatomic, strong) MITemporaryImage *outputTempImage;

//- (instancetype)initWithOutputShape:(DataShape *)outputShape;
- (DataShape *)outputShapeRef;

- (void)compile:(id<MTLDevice>)device;

- (void)removeOutputTempImage;
- (void)setOutputTempImageToTargets;
- (void)notifyTargetsAboutNewTempImage:(id<MTLCommandBuffer>)cmdBuf;

- (NSArray<id<MetalTensorInput>>*)targets;

- (void)addTarget:(id<MetalTensorInput>)newTarget;
- (void)addTarget:(id<MetalTensorInput>)newTarget atTempImageIndex:(NSInteger)imageIndex;

- (void)removeTarget:(id<MetalTensorInput>)targetToRemove;
- (void)removeAllTargets;

//#ifdef DEBUG
@property (nonatomic, assign) int verbose;
- (const char*)labelUTF8;
//#endif

@end

NS_ASSUME_NONNULL_END
