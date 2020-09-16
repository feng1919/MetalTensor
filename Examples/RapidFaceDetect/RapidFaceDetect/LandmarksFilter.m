//
//  LandmarksFilter.m
//  RapidFaceDetect
//
//  Created by Feng Stone on 2020/7/25.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import "LandmarksFilter.h"
#import "LandmarksNet.h"
#import "RapidFaceDetectNet.h"
#import "CropFilter.h"

@interface LandmarksFilter() <RapidFaceDetectNetDelegate, LandmarksNetDelegate> {
    
    RapidFaceDetectNet *_faceNet;
    CropFilter *_faceCrop;
    
    LandmarksNet *_landmarksNet;
    CropFilter *_landmarksCrop;
    
    MTLUInt2 _texture_size;
    CGRect _faceRect;
    
    dispatch_semaphore_t _semaphore;
    MetalImageTexture *_processingTexture;
}

@end

@implementation LandmarksFilter

- (id)init {
    if (self = [super init]) {
        _semaphore = dispatch_semaphore_create(1);
        _faceRect = CGRectZero;
    }
    
    return self;
}

- (void)createNet {
    
    id<MTLDevice> device = [MetalDevice sharedMTLDevice];
    
    _faceNet = [[RapidFaceDetectNet alloc] init];
    [_faceNet setSynchronizedProcessing:NO];
    [_faceNet compile:device];
    [_faceNet loadWeights];
    [_faceNet setDelegate:self];
    
    MetalImageFilter *resizeFilter = [[MetalImageFilter alloc] init];
    resizeFilter.outputImageSize = _faceNet.inputSize;
    
    _faceCrop = [[CropFilter alloc] init];
    [_faceCrop setFramebuffer:YES];
    [_faceCrop addTarget:resizeFilter];
    [resizeFilter addTarget:_faceNet];
    
    _landmarksNet = [[LandmarksNet alloc] init];
    [_landmarksNet setSynchronizedProcessing:NO];
    [_landmarksNet compile:device];
    [_landmarksNet loadWeights];
    [_landmarksNet setDelegate:self];
    
    MetalImageFilter *resizeFilter1 = [[MetalImageFilter alloc] init];
    resizeFilter1.outputImageSize = _landmarksNet.inputSize;
    
    _landmarksCrop = [[CropFilter alloc] init];
    _landmarksCrop.bgClearColor = MTLClearColorMake(0.5f, 0.5f, 0.5f, 1.0f);
    _landmarksCrop.framebuffer = NO;
//    _landmarksCrop.outputImageSize = _landmarksNet.inputSize;
    [_landmarksCrop addTarget:resizeFilter1];
    [resizeFilter1 addTarget:_landmarksNet];
    
}

- (void)updateCropCenter {
    
    float width = _texture_size.x;
    float height = _texture_size.y;
    float size = MIN(_texture_size.x, _texture_size.y);
    
    float x, y;
    if (CGRectIsEmpty(_faceRect)) {
        x = (width-size)*0.5f;
        y = (height-size)*0.5f;
    }
    else {
        x = MAX(MIN(CGRectGetMidX(_faceRect)-size*0.5f, width-size), 0.0f);
        y = MAX(MIN(CGRectGetMidY(_faceRect)-size*0.5f, height-size), 0.0f);
    }
    
    _faceCrop.cropRegion = CGRectMake(x/width, y/height, size/width, size/height);
//    NSLog(@"change crop region: %@", NSStringFromCGRect(_faceCrop.cropRegion));
}

#pragma mark - MetalImageInput
- (void)setInputTexture:(MetalImageTexture *)newInputTexture atIndex:(NSInteger)textureIndex {
    NSParameterAssert(newInputTexture != nil);
    
    if (firstInputTexture) {
        NSLog(@"Landmark Detecting Dropped One Frame.");
        [firstInputTexture unlock];
    }
    
    firstInputTexture = newInputTexture;
    [firstInputTexture lock];
    
    if (MTLUInt2Equal(_texture_size, firstInputTexture.size) == NO) {
        _texture_size = firstInputTexture.size;
        [self updateCropCenter];
    }
}

- (void)newTextureReadyAtTime:(CMTime)frameTime atIndex:(NSInteger)textureIndex {
    firstInputParameter.frameTime = frameTime;
    
    if (dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_NOW) != 0) {
        NSLog(@"processing wait...");
        return;
    }
    
    _processingTexture = firstInputTexture;
    [_processingTexture lock];
    
    [firstInputTexture unlock];
    firstInputTexture = nil;
    
    [_faceCrop setInputTexture:_processingTexture atIndex:0];
    [_faceCrop newTextureReadyAtTime:frameTime atIndex:0];
}

- (void)renderToTexture:(MetalImageTexture *)texture {
    
    [self updateTextureVertexBuffer:_verticsBuffer
                    withNewContents:MetalImageDefaultRenderVetics
                               size:MetalImageDefaultRenderVetexCount];
    [self updateTextureCoordinateBuffer:_coordBuffer
                        withNewContents:TextureCoordinatesForRotation(firstInputParameter.rotationMode)
                                   size:MetalImageDefaultRenderVetexCount];
    
    id<MTLCommandBuffer> commandBuffer = [MetalDevice sharedCommandBuffer];

    outputTexture = [[MetalImageContext sharedTextureCache] fetchTextureWithSize:[self textureSizeForOutput]];
    NSParameterAssert(outputTexture);
    
    MTLRenderPassColorAttachmentDescriptor *colorAttachment = _renderPassDescriptor.colorAttachments[0];
    colorAttachment.texture = [outputTexture texture];
    
    id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:_renderPassDescriptor];
    NSAssert(renderEncoder != nil, @"Create render encoder failed...");
    [renderEncoder setDepthStencilState:_depthStencilState];
    [renderEncoder setRenderPipelineState:_pipelineState];
    [renderEncoder setVertexBuffer:_verticsBuffer offset:0 atIndex:0];
    [renderEncoder setVertexBuffer:_coordBuffer offset:0 atIndex:1];
    [renderEncoder setFragmentTexture:[texture texture] atIndex:0];
    [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:MetalImageDefaultRenderVetexCount instanceCount:1];
    [renderEncoder endEncoding];
    [texture unlock];
}

- (MTLUInt2)textureSizeForOutput {
    return _texture_size;
}

#pragma mark - RapidFaceDetectNet delegate

- (void)RapidFaceDetectNet:(RapidFaceDetectNet *)net didFinishWithObjects:(NSArray<SSDObject *> *)objects1 {
    
    if (objects1.count == 0) {
        _faceRect = CGRectZero;
        
        [self renderToTexture:_processingTexture];
        [self notifyTargetsAboutNewTextureAtTime:firstInputParameter.frameTime];
        dispatch_semaphore_signal(_semaphore);
        
        return;
    }
    
    NSArray<SSDObject *> *objects = [objects1 copy];
    
    __weak __auto_type ws = self;
    
    runMetalAsynchronouslyOnVideoProcessingQueue(^{
        
        __strong __auto_type ss = ws;
        
        float width = ss->_texture_size.x;
        float height = ss->_texture_size.y;
        float size = MIN(width, height);
        float x = ss->_faceCrop.cropRegion.origin.x * width;
        float y = ss->_faceCrop.cropRegion.origin.y * height;
        
        SSDObject *face = objects.firstObject;
        float xmin = face.xmin * size + x;
        float xmax = face.xmax * size + x;
        float ymin = face.ymin * size + y;
        float ymax = face.ymax * size + y;
        
        ss->_faceRect = CGRectMake(xmin, ymin, xmax-xmin, ymax-ymin);
        [ss updateCropCenter];
        
        float w = xmax - xmin;
        float h = ymax - ymin;
        float s = fmaxf(w, h) * 1.1f * 0.5f;
        float xmid = (xmax + xmin) * 0.5f;
        float ymid = (ymax + ymin) * 0.5f;
        xmin = xmid - s;
        xmax = xmid + s;
        ymin = ymid - s;
        ymax = ymid + s;
        
        ss->_landmarksCrop.cropRegion = CGRectMake(xmin / width, ymin / height, (xmax - xmin) / width, (ymax - ymin) / height);
        [ss->_landmarksCrop setInputTexture:ss->_processingTexture atIndex:0];
        [ss->_landmarksCrop newTextureReadyAtTime:ss->firstInputParameter.frameTime atIndex:0];
        
    });
}

#pragma mark - LandmarksFilter delegate

- (void)LandmarksNet:(LandmarksNet *)net didFinishWithPoints:(float32_t *)points1 {
    
    float32_t *points = malloc(sizeof(float32_t) * 136);
    float x = _landmarksCrop.cropRegion.origin.x * _texture_size.x;
    float y = _landmarksCrop.cropRegion.origin.y * _texture_size.y;
    float width = _landmarksCrop.cropRegion.size.width * _texture_size.x;
    float height = _landmarksCrop.cropRegion.size.height * _texture_size.y;
    
    for (int i = 0; i < 136; i=i+2) {
        points[i] = points1[i] / 96.0f * width + x;
        points[i+1] = points1[i+1] / 96.0f * height + y;
    }
    
    __weak __auto_type ws = self;
    runMetalAsynchronouslyOnVideoProcessingQueue(^{
        
        __strong __auto_type ss = ws;
        
        int width = ss->_texture_size.x;
        int height = ss->_texture_size.y;
        
        MetalImageTexture *texture = ss->_processingTexture;
        Byte *buffer = [texture byteBuffer];
        NSUInteger bpr = [texture bytesPerRow];
        static CGColorSpaceRef colorSpace = NULL;
        if (colorSpace == NULL) {
            colorSpace = CGColorSpaceCreateDeviceRGB();
        }
        
        CGContextRef context = CGBitmapContextCreate(buffer, width, height, 8, bpr, colorSpace, kCGImageByteOrder32Little|kCGImageAlphaPremultipliedLast);
        CGContextSaveGState(context);
        CGContextSetStrokeColorWithColor(context, [UIColor colorWithRed:1.0f green:1.0f blue:1.0f alpha:0.5f].CGColor);
        CGContextSetLineWidth(context, 5.0f);
        CGContextAddRect(context, CGRectMake(ss->_faceRect.origin.x, height-CGRectGetMaxY(ss->_faceRect),
                                             CGRectGetWidth(ss->_faceRect), CGRectGetHeight(ss->_faceRect)));
        CGContextStrokePath(context);
        CGContextSetFillColorWithColor(context, [UIColor whiteColor].CGColor);
        for (int i = 0; i < 68; i++) {
            CGContextFillEllipseInRect(context, CGRectMake(points[i*2]-3, height-(points[i*2+1])-3, 7, 7));
        }
        CGContextRestoreGState(context);
        CGContextRelease(context);
        free(points);

        [self renderToTexture:ss->_processingTexture];
        [self notifyTargetsAboutNewTextureAtTime:ss->firstInputParameter.frameTime];
        
        dispatch_semaphore_signal(ss->_semaphore);
    });
}

@end
