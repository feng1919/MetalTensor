//
//  MetalTensorInputLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/5.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorInputLayer.h"
#import <MetalImage/MetalDevice.h>

@implementation MetalTensorInputLayer

- (instancetype)initWithOutputShape:(DataShape *)outputShape {
    if (self = [super initWithOutputShape:outputShape]) {
        _outputImage = [[MIMPSImage alloc] initWithShape:&_outputShape];
        [_outputImage setReferenceCountingEnable:NO];
        
        MPSImageDescriptor *descriptor = ImageDescriptor(outputShape);
        descriptor.storageMode = MTLStorageModeShared;
        _outputImage.mpsImage = [[MPSImage alloc] initWithDevice:[MetalDevice sharedMTLDevice] imageDescriptor:descriptor];

        DB_TRACE(-_verbose+2, "\n%s init --> %s", self.labelUTF8, NSStringFromDataShape(outputShape).UTF8String);
    }
    return self;
}

- (void)inputTexture:(id<MTLTexture>)bgraU8Texture {
    DB_TRACE(-_verbose+2, "\n\n\n%s <-- (%ld, %ld, 4)", self.labelUTF8, bgraU8Texture.width, bgraU8Texture.height);
    _outputImage.mpsImage = [[MPSImage alloc] initWithTexture:bgraU8Texture featureChannels:3];
}

- (void)processOnCommandBuffer:(nonnull id<MTLCommandBuffer>)cmdBuf {

    for (id<MetalTensorInput> currentTarget in _targets) {
        NSInteger indexOfObject = [_targets indexOfObject:currentTarget];
        NSInteger tempImageIndex = [_targetTempImageIndices[indexOfObject] integerValue];
        
        [currentTarget setInputImage:_outputImage atIndex:tempImageIndex];
        
        DB_TRACE(-_verbose+1, "\n%s ---%s---> %s(%ld)", self.labelUTF8,
                 NSStringFromDataShape(_outputImage.shape).UTF8String,
                 [currentTarget description].UTF8String, tempImageIndex);
        
        [currentTarget tempImageReadyAtIndex:tempImageIndex commandBuffer:cmdBuf];
    }
}

@end