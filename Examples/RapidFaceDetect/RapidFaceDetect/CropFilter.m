//
//  CropFilter.m
//  RapidFaceDetect
//
//  Created by Feng Stone on 2020/7/26.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import "CropFilter.h"

@implementation CropFilter

- (void)setInputTexture:(MetalImageTexture *)newInputTexture atIndex:(NSInteger)textureIndex {
    [super setInputTexture:newInputTexture atIndex:textureIndex];
    
}

- (void)renderToTextureWithVertices:(const MTLFloat4 *)vertices textureCoordinates:(const MTLFloat2 *)textureCoordinates {
    NSParameterAssert(vertices);
    NSParameterAssert(textureCoordinates);
    
    [self updateTextureVertexBuffer:_verticsBuffer withNewContents:vertices size:MetalImageDefaultRenderVetexCount];
    [self updateTextureCoordinateBuffer:_coordBuffer withNewContents:textureCoordinates size:MetalImageDefaultRenderVetexCount];
    
    id<MTLCommandBuffer> commandBuffer = [MetalDevice sharedCommandBuffer];

    if (self.framebuffer) {
        outputTexture = [[MetalImageContext sharedTextureCache] fetchTextureWithSize:[self textureSizeForOutput]];
    }
    else {
        outputTexture = [[MetalImageTexture alloc] initWithTextureSize:[self textureSizeForOutput]
                                                          textureUsage:MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead];
    }
    NSParameterAssert(outputTexture);
    
    MTLRenderPassColorAttachmentDescriptor *colorAttachment = _renderPassDescriptor.colorAttachments[0];
    colorAttachment.texture = [outputTexture texture];
    
    id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:_renderPassDescriptor];
    NSAssert(renderEncoder != nil, @"Create render encoder failed...");
    [self assembleRenderEncoder:renderEncoder];
    
    [firstInputTexture unlock];
}

@end
