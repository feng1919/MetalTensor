//
//  MITemporaryImage.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/19.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MPSImage+Extension.h"
#import "metal_tensor_structures.h"

NS_ASSUME_NONNULL_BEGIN

@interface MITemporaryImage : NSObject

@property (nonatomic, strong, nullable) MPSTemporaryImage *image;
@property (nonatomic, assign) BOOL referenceCountingEnable; //default YES
@property (nonatomic, assign) NSInteger reuseIdentifier;

- (instancetype)initWithShape:(DataShape *)image;

- (DataShape *)shape;
- (MPSImageFeatureChannelFormat)channelFormat;
- (NSUInteger)width;
- (NSUInteger)height;
- (NSUInteger)depth;

- (MPSImageDescriptor *)imageDescriptor;
- (MPSTemporaryImage *)newTemporaryImageForCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
- (void)deleteTemporaryImage;

- (void)lock;
- (void)unlock;
- (int)referenceCounting;

MPSImageDescriptor *ImageDescriptor(DataShape *s);

@end

NS_ASSUME_NONNULL_END
