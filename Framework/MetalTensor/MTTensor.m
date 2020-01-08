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
    int _referenceCounting;
    MPSImageDescriptor *_imageDescriptor;
    MPSTemporaryImage *_image;
}

@end

MPSImageDescriptor *ImageDescriptor(DataShape *s) {
    assert(s->row>0 && s->column>0 && s->depth>0);
    MPSImageDescriptor *desc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                                                              width:s->column
                                                                             height:s->row
                                                                    featureChannels:s->depth];
    desc.storageMode = MTLStorageModePrivate;
    return desc;
}

@implementation MTTensor

- (instancetype)initWithShape:(DataShape *)image {
    if (self = [super init]) {
        _shape = image[0];
        _referenceCounting = 0;
        _imageDescriptor = ImageDescriptor(image);
        _referenceCountingEnable = YES;
    }
    return self;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"Tensor: %@", KeyForTensorType(&_shape)];
}

- (void)dealloc {
    
    NSAssert(_referenceCountingEnable == NO || _referenceCounting == 0, @"Unexpected dealloc...");
    
    self.image = nil;
    NSLog(@"Temporary image dealloc: %@", KeyForTensorType(&_shape));
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

- (MPSTemporaryImage *)newContentOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    @synchronized (self) {
        if (!_image) {
//        _image.readCount = 0;
            _image = [MPSTemporaryImage temporaryImageWithCommandBuffer:commandBuffer imageDescriptor:_imageDescriptor];
            _image.readCount = NSIntegerMax;
        }
        return _image;
    }
}

- (void)deleteContent {
    self.image = nil;
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
            [[MTTensorCache sharedCache] cacheTensor:self];
        }
    }
}

- (int)referenceCounting {
    return _referenceCounting;
}

@end
