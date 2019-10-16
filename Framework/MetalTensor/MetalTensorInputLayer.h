//
//  MetalTensorInputLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/5.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorNode.h"
#import "MIMPSImage.h"

NS_ASSUME_NONNULL_BEGIN

@interface MetalTensorInputLayer : MetalTensorNode

@property (nonatomic, strong) MIMPSImage *outputImage;

- (instancetype)initWithInputShape:(DataShape *)inputShape;

- (void)inputTexture:(id<MTLTexture>)bgraU8Texture;
- (void)processOnCommandBuffer:(id<MTLCommandBuffer>)cmdBuf;

@end

NS_ASSUME_NONNULL_END
