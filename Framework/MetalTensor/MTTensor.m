//
//  MTTensor.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/19.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MTTensor.h"
#import "MTTensorCache.h"

@interface MTTensor() {
    DataShape _shape;
    MPSImageDescriptor *_imageDescriptor;
    MPSTemporaryImage *_image;
}

@end

@implementation MTTensor

@synthesize poolIdentifier = _poolIdentifier;
@synthesize referenceCountingEnable = _referenceCountingEnable;
@synthesize referenceCounting = _referenceCounting;

- (instancetype)init{
    NSAssert(NO, @"Invalid initialize function.");
    return nil;
}

- (instancetype)initWithShape:(DataShape *)shape {
    if (self = [super init]) {
        _dataFormat = MPSDataTypeFloat16;
        _shape = shape[0];
        _referenceCounting = 0;
        _imageDescriptor = ImageDescriptor(shape, _dataFormat);
        _referenceCountingEnable = YES;
    }
    return self;
}

- (instancetype)initWithShape:(DataShape *)shape dataType:(MPSDataType)dataType numberOfImage:(NSUInteger)numberOfImages {
    if (self = [super init]) {
        _dataFormat = dataType;
        _shape = shape[0];
        _referenceCounting = 0;
        _imageDescriptor = ImageDescriptor(shape, _dataFormat);
        _imageDescriptor.numberOfImages = numberOfImages;
        _referenceCountingEnable = YES;
    }
    return self;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"Tensor: %@", KeyForTensorType1(&_shape, _dataFormat, _imageDescriptor.numberOfImages)];
}

- (void)dealloc {
    
    NSAssert(_referenceCountingEnable == NO || _referenceCounting == 0, @"Unexpected dealloc...");
    
    self.image = nil;
    
#if DEBUG
    if (_referenceCountingEnable) {
        NSLog(@"Temporary image dealloc: %@", KeyForTensorType(&_shape, _dataFormat));
    }
#endif
}

#pragma mark - Public Access

- (MPSImage *)content {
    return _image;
}

- (DataShape *)shape {
    return &_shape;
}

- (MPSImageFeatureChannelFormat)channelFormat {
    return MPSImageFeatureChannelFormatFloat16;
}

- (NSUInteger)width {
    return _shape.column;
}

- (NSUInteger)height {
    return _shape.row;
}

- (NSUInteger)depth {
    return _shape.depth;
}

- (MPSImageDescriptor *)imageDescriptor {
    return _imageDescriptor;
}

- (void)setImage:(MPSTemporaryImage *)image {
    NSAssert(image==nil||[image isKindOfClass:[MPSTemporaryImage class]], @"Invalid image type...");
    @synchronized (self) {
        if (_image.readCount > 0) {
            _image.readCount = 0;
        }
        _image = image;
    }
}

#pragma mark - Reference Counting

- (void)lock {
    if (_referenceCountingEnable) {
        _referenceCounting++;
    }
}

- (void)unlock {
    if (_referenceCountingEnable) {
        NSAssert(_referenceCounting > 0, @"Tried to overrelease a temporary image.");
        _referenceCounting--;
        if (_referenceCounting < 1) {
            //        if ([_image isKindOfClass:[MPSTemporaryImage class]]) {
            //            [(MPSTemporaryImage *)_image setReadCount:0];
            //        }
            //        self.image = nil;
            [[MTTensorCache sharedCache] cacheResource:self];
        }
    }
}

- (int)referenceCounting {
    return _referenceCounting;
}

- (void)newContentOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    @synchronized (self) {
        if (!_image) {
//        _image.readCount = 0;
            _image = [MPSTemporaryImage temporaryImageWithCommandBuffer:commandBuffer imageDescriptor:_imageDescriptor];
            _image.readCount = NSIntegerMax;
        }
    }
}

- (void)deleteContent {
    self.image = nil;
}

- (NSString *)reuseIdentifier {
    return KeyForTensorType(self.shape, self.dataFormat);
}

@end

MPSImageDescriptor *ImageDescriptor(DataShape *s, MPSDataType dataFormat) {
    assert(s->row>0 && s->column>0 && s->depth>0);
    MPSImageFeatureChannelFormat format = dataFormat == MPSDataTypeFloat16?MPSImageFeatureChannelFormatFloat16:MPSImageFeatureChannelFormatFloat32;
    MPSImageDescriptor *desc = [MPSImageDescriptor imageDescriptorWithChannelFormat:format
                                                                              width:s->column
                                                                             height:s->row
                                                                    featureChannels:s->depth];
    desc.storageMode = MTLStorageModePrivate;
    return desc;
}

NSString *KeyForTensorType(DataShape *shape, MPSDataType dataFormat) {
    return KeyForTensorType1(shape, dataFormat, 1);
}

NSString *KeyForTensorType1(DataShape *shape, MPSDataType dataFormat, NSUInteger numberOfImages) {
//    return [NSString stringWithFormat:@"[ROW %d][COLUMN %d][DEPTH %d][FLOAT %d][N%d]", shape->row, shape->column, shape->depth, dataFormat, (int)numberOfImages];
    return [NSString stringWithFormat:@"<NHWC: %dx%dx%dx%d - float%d>", (int)numberOfImages,
            shape->row, shape->column, shape->depth, dataFormat==MPSDataTypeFloat16?16:32];
}
