//
//  PortraitSegmentFilter.m
//  MetalTensorDemo
//
//  Created by Feng Stone on 2019/9/27.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "PortraitSegmentFilter.h"
#import "PortraitSegmentNet.h"

@interface PortraitSegmentFilter () {
    
    id<MTLTexture> _mask_texture;
    MTLUInt2 _texture_size;
    MTLUInt2 _mask_size;
    
    dispatch_semaphore_t _semaphore;
    MetalImageTexture *_processingTexture;
    
}

@property (nonatomic, strong) MetalImageFilter *resizeFilter;
@property (nonatomic, strong) PortraitSegmentNet *segmentNet;

@end

@implementation PortraitSegmentFilter

- (id)init
{
    id<MTLLibrary> library = [MetalDevice.sharedMTLDevice newDefaultLibrary];
    id<MTLFunction> fragment = [library newFunctionWithName:@"fragment_portraitRender"];
    if (self = [super initWithFragmentFunction:fragment]) {
        _semaphore = dispatch_semaphore_create(1);
    }
    
    return self;
}

- (void)createNet {
    PortraitSegmentNet *net = [[PortraitSegmentNet alloc] init];
    [net setSynchronizedProcessing:NO];
    [net compile:[MetalDevice sharedMTLDevice]];
    [net loadWeights];
    [net setDelegate:self];
    self.segmentNet = net;
    
    MetalImageFilter *resize = [[MetalImageFilter alloc] init];
    resize.outputImageSize = net.inputSize;
    self.resizeFilter = resize;
    
    [self.resizeFilter addTarget:self.segmentNet];
    
    _mask_size = net.outputMaskSize;
    
    id<MTLDevice> device = [MetalDevice sharedMTLDevice];
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                                                          width:_mask_size.x
                                                                                         height:_mask_size.y
                                                                                      mipmapped:NO];
    _mask_texture = [device newTextureWithDescriptor:descriptor];
}

- (void)renderToTexture {
    
    [self updateTextureVertexBuffer:_verticsBuffer withNewContents:MetalImageDefaultRenderVetics size:MetalImageDefaultRenderVetexCount];
    [self updateTextureCoordinateBuffer:_coordBuffer withNewContents:TextureCoordinatesForRotation(firstInputParameter.rotationMode) size:MetalImageDefaultRenderVetexCount];
    
    id<MTLCommandBuffer> commandBuffer = [MetalDevice sharedCommandBuffer];

    outputTexture = [[MetalImageContext sharedTextureCache] fetchTextureWithSize:[self textureSizeForOutput]];
    NSParameterAssert(outputTexture);
    
    MTLRenderPassColorAttachmentDescriptor *colorAttachment = _renderPassDescriptor.colorAttachments[0];
    colorAttachment.texture = [outputTexture texture];
    
    id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:_renderPassDescriptor];
    NSAssert(renderEncoder != nil, @"Create render encoder failed...");
    [self assembleRenderEncoder:renderEncoder];
    
    [_processingTexture unlock];
    _processingTexture = nil;
}

- (void)assembleRenderEncoder:(id<MTLRenderCommandEncoder>)renderEncoder {
    
    NSParameterAssert(renderEncoder);
    
    [renderEncoder setDepthStencilState:_depthStencilState];
    [renderEncoder setRenderPipelineState:_pipelineState];
    [renderEncoder setVertexBuffer:_verticsBuffer offset:0 atIndex:0];
    [renderEncoder setVertexBuffer:_coordBuffer offset:0 atIndex:1];
    [renderEncoder setFragmentTexture:[_processingTexture texture] atIndex:0];
    [renderEncoder setFragmentTexture:_mask_texture atIndex:1];
    [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0
                      vertexCount:MetalImageDefaultRenderVetexCount instanceCount:1];
    [renderEncoder endEncoding];
}

- (MTLUInt2)textureSizeForOutput {
    return _texture_size;
}

#pragma mark - MetalImageInput
- (void)setInputTexture:(MetalImageTexture *)newInputTexture atIndex:(NSInteger)textureIndex {
    NSParameterAssert(newInputTexture != nil);
    
    if (firstInputTexture) {
        NSLog(@"Portrait Render Drop One Frame.");
        [firstInputTexture unlock];
    }
    firstInputTexture = newInputTexture;
    [firstInputTexture lock];
    _texture_size = firstInputTexture.size;
}

- (void)newTextureReadyAtTime:(CMTime)frameTime atIndex:(NSInteger)textureIndex {
    firstInputParameter.frameTime = frameTime;
    
    if (dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_NOW) != 0) {
        return;
    }
    
    _processingTexture = firstInputTexture;
    firstInputTexture = nil;
    
    [_resizeFilter setInputTexture:_processingTexture atIndex:0];
    [_resizeFilter newTextureReadyAtTime:frameTime atIndex:0];
}

#pragma mark - PortraitSegmentNet delegate
- (void)PortraitSegmentNet:(PortraitSegmentNet *)net predictResult:(float16_t *)result
{
    runMetalAsynchronouslyOnVideoProcessingQueue(^{
        
        [_mask_texture replaceRegion:MTLRegionMake2D(0, 0, _mask_size.x, _mask_size.y)
                         mipmapLevel:0
                           withBytes:(void *)result
                         bytesPerRow:_mask_size.x*sizeof(float16_t)];
        
        [self renderToTexture];
        [self notifyTargetsAboutNewTextureAtTime:firstInputParameter.frameTime];
        
        dispatch_semaphore_signal(_semaphore);
        
        if (firstInputTexture != nil) {
            [self newTextureReadyAtTime:firstInputParameter.frameTime atIndex:0];
        }
    });
}

@end
