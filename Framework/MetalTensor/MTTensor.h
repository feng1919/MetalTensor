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
#import "MTResourceProtocol.h"

NS_ASSUME_NONNULL_BEGIN

@protocol MTBackwardDelegate;
@interface MTTensor : NSObject <MTResource>

@property (nonatomic, weak, nullable) id<MTBackwardDelegate> source;

@property (nonatomic, readonly) MPSDataType dataFormat;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithShape:(DataShape *)shape NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithShape:(DataShape *)shape dataType:(MPSDataType)dataType numberOfImage:(NSUInteger)numberOfImages NS_DESIGNATED_INITIALIZER;

- (MPSImage *)content;
- (DataShape *)shape;
- (MPSImageFeatureChannelFormat)channelFormat;
- (NSUInteger)width;
- (NSUInteger)height;
- (NSUInteger)depth;

- (MPSImageDescriptor *)imageDescriptor;

@end

typedef MTTensor * MetalTensor;

MPSImageDescriptor *ImageDescriptor(DataShape *s, MPSDataType);
NSString *KeyForTensorType(DataShape *shape, MPSDataType dataFormat);
NSString *KeyForTensorType1(DataShape *shape, MPSDataType dataFormat, NSUInteger numberOfImages);

NS_ASSUME_NONNULL_END
