//
//  MTTensor.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/19.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MPSImage+Extension.h"
#import "metal_tensor_structures.h"

NS_ASSUME_NONNULL_BEGIN

@protocol MTBackwardDelegate;
@interface MTTensor : NSObject

@property (nonatomic, assign) BOOL referenceCountingEnable; //default YES
@property (nonatomic, assign) NSInteger reuseIdentifier;

@property (nonatomic, weak, nullable) id<MTBackwardDelegate> source;

- (instancetype)initWithShape:(DataShape *)image;

- (MPSImage *)content;
- (DataShape *)shape;
- (MPSImageFeatureChannelFormat)channelFormat;
- (NSUInteger)width;
- (NSUInteger)height;
- (NSUInteger)depth;

- (MPSImageDescriptor *)imageDescriptor;
- (MPSTemporaryImage *)newContentOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
- (void)deleteContent;

- (void)lock;
- (void)unlock;
- (int)referenceCounting;

MPSImageDescriptor *ImageDescriptor(DataShape *s);

@end

typedef MTTensor * MetalTensor;

NS_ASSUME_NONNULL_END