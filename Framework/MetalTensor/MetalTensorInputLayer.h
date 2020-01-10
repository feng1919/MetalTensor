//
//  MetalTensorInputLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/5.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"
#import "MTImageTensor.h"

NS_ASSUME_NONNULL_BEGIN

@interface MetalTensorInputLayer : MetalTensorLayer

/*
 *  RGB
 *  3 channels
 */
@property (nonatomic, strong) MTImageTensor *outputImage;

@property (nonatomic, strong) MTImageTensor *gradientImage;

- (void)inputTexture:(id<MTLTexture>)bgraU8Texture;

@end

NS_ASSUME_NONNULL_END
