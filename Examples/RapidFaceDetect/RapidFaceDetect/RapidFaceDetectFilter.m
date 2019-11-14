//
//  RapidFaceDetectFilter.m
//  RapidFaceDetect
//
//  Created by Feng Stone on 2019/11/13.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "RapidFaceDetectFilter.h"
#import "RapidFaceDetectNet.h"

// render 20 objects at most.
const int MAX_NUM_OF_OBJECTS = 20;

@interface RapidFaceDetectFilter () <RapidFaceDetectNetDelegate> {
    
    RapidFaceDetectNet *_net;
    MetalImageFilter *_lens;
    MTLUInt2 _texture_size;
    
    dispatch_semaphore_t _semaphore;
    MetalImageTexture *_processingTexture;
}

@property (nonatomic,strong) id<MTLBuffer> buffer;
@property (nonatomic,strong) id<MTLBuffer> buffer_count;

@end

@implementation RapidFaceDetectFilter

- (id)init {
    id<MTLLibrary> library = [MetalDevice.sharedMTLDevice newDefaultLibrary];
    id<MTLFunction> fragment = [library newFunctionWithName:@"fragment_object_detect_net_render"];
    if (self = [super initWithFragmentFunction:fragment]) {
        _semaphore = dispatch_semaphore_create(1);
    }
    
    return self;
}

- (void)dealloc {
    _buffer = nil;
    _buffer_count = nil;
}

- (void)createNet {
    
    id<MTLDevice> device = [MetalDevice sharedMTLDevice];
    _buffer = [device newBufferWithLength:sizeof(MTLFloat4)*MAX_NUM_OF_OBJECTS options:MTLResourceOptionCPUCacheModeDefault];
    _buffer_count = [device newBufferWithLength:sizeof(MTLUInt) options:MTLResourceOptionCPUCacheModeDefault];
    
    _net = [[RapidFaceDetectNet alloc] init];
    [_net setSynchronizedProcessing:NO];
    [_net compile:device];
    [_net loadWeights];
    [_net setDelegate:self];
    
    _lens = [[MetalImageFilter alloc] init];
    _lens.outputImageSize = _net.inputSize;
    [_lens addTarget:_net];
}

#pragma mark - Render to target

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
    [renderEncoder setFragmentBuffer:_buffer offset:0 atIndex:0];
    [renderEncoder setFragmentBuffer:_buffer_count offset:0 atIndex:1];
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
        NSLog(@"Rapid Face Detecting Dropped One Frame.");
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
    
    [_lens setInputTexture:_processingTexture atIndex:0];
    [_lens newTextureReadyAtTime:frameTime atIndex:0];
}

#pragma mark - RapidFaceDetectNet delegate

- (void)RapidFaceDetectNet:(RapidFaceDetectNet *)net didFinishWithObjects:(NSArray<SSDObject *> *)objects1 {
    
    NSArray<SSDObject *> *objects = [objects1 copy];
    
    __weak __auto_type ws = self;
    
    runMetalAsynchronouslyOnVideoProcessingQueue(^{
        
        __strong __auto_type ss = ws;
        
        int count = (int)objects.count;
        MTLUInt *bufferCount = (MTLUInt *)[ss->_buffer_count contents];
        bufferCount[0] = count;
        MTLFloat4 *bufferContents = (MTLFloat4 *)[ss->_buffer contents];
        
        for (int i = 0; i < MIN(MAX_NUM_OF_OBJECTS, count); i++) {
            bufferContents[i].x = objects[i].xmin;
            bufferContents[i].y = objects[i].xmax;
            bufferContents[i].z = objects[i].ymin;
            bufferContents[i].w = objects[i].ymax;
        }
        
        [ss renderToTexture];
        [ss notifyTargetsAboutNewTextureAtTime:ss->firstInputParameter.frameTime];
        
        dispatch_semaphore_signal(ss->_semaphore);
        
        if (ss->firstInputTexture != nil) {
            [ss newTextureReadyAtTime:ss->firstInputParameter.frameTime atIndex:0];
        }
    });
}

@end
