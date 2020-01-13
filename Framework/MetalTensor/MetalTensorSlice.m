//
//  MetalTensorSlice.m
//  MetalTensor
//
//  Created by Feng Stone on 2020/1/11.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import "MetalTensorSlice.h"
#import "MTTensorCache.h"

@interface SliceDataSource : NSObject <MPSCNNConvolutionDataSource> {
    
    int _numberOfChannel;
    MPSCNNConvolutionDescriptor *_convDesc;
    float32_t *_data;
}

@end

@implementation SliceDataSource

- (id)copyWithZone:(NSZone *)zone {
    SliceDataSource *item = [SliceDataSource allocWithZone:zone];
    item->_numberOfChannel = _numberOfChannel;
    item->_data = _data;
    item->_convDesc = _convDesc;
    return item;
}

- (id)copy {
    SliceDataSource *item = [[SliceDataSource alloc] initWithNumberOfChannel:_numberOfChannel];
    item->_data = _data;
    item->_convDesc = _convDesc;
    return item;
}

- (instancetype)initWithNumberOfChannel:(int)numberOfChannel {
    if (self = [super init]) {
        _numberOfChannel = numberOfChannel;
        _convDesc = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:1
                                                                            kernelHeight:1
                                                                    inputFeatureChannels:numberOfChannel
                                                                   outputFeatureChannels:4];
        _data = calloc(_numberOfChannel * 4, sizeof(float32_t));
    }
    return self;
}

- (void)activeChannelAtIndex:(int)channelIndex {
    for (int i = 0; i < _numberOfChannel; i++) {
        for (int j = 0; j < 4; j ++) {
            _data[i+j*_numberOfChannel] = i == channelIndex?1.0f:0.0f;
        }
    }
}

- (void)dealloc {
    free(_data);
}

- (MPSDataType)dataType {
    return MPSDataTypeFloat32;
}

- (void *)weights {
    return _data;
}

- (float32_t *)biasTerms {
    return NULL;
}

- (BOOL)load {
    return YES;
}

- (void)purge {
}

- (NSString *)label {
    return [NSString stringWithFormat:@"SliceDataSource: %d", _numberOfChannel];
}

- (MPSCNNConvolutionDescriptor *)descriptor {
    return _convDesc;
}

@end

@interface MetalTensorSlice() {
    
    int _numberOfChannel;
    SliceDataSource *_dataSource;
    MPSCNNConvolution *_convolution;
    id<MTLDevice> _device;
}

@end

@implementation MetalTensorSlice
- (instancetype)initWithNumberOfChannel:(int)numberOfChannel {
    if (self = [super init]) {
        _numberOfChannel = numberOfChannel;
        _dataSource = [[SliceDataSource alloc] initWithNumberOfChannel:numberOfChannel];
    }
    return self;
}

- (void)compile:(id<MTLDevice>)device {
    _device = device;
    _convolution = [[MPSCNNConvolution alloc] initWithDevice:device weights:_dataSource];
}

- (MetalTensor)sliceTensor:(MetalTensor)src channelIndex:(int)channelIndex commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    NSAssert(src.shape->depth == _numberOfChannel, @"The input tensor's depth is not identical to the initialized.");
    NSAssert(channelIndex < _numberOfChannel, @"Invalid channel index");
    
    DataShape dstShape = DataShapeMake(src.shape->row, src.shape->column, 4);
    MetalTensor dst = [[MTTensorCache sharedCache] fetchTensorWithShape:&dstShape commandBuffer:commandBuffer];
    
    [_dataSource activeChannelAtIndex:channelIndex];
    [_convolution reloadWeightsAndBiasesFromDataSource];
    [_convolution encodeToCommandBuffer:commandBuffer sourceImage:src.content destinationImage:dst.content];
    return dst;
}

- (void)sliceTensor:(MetalTensor)src toTensor:(MetalTensor)dst channelIndex:(int)channelIndex commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    NSAssert(src.shape->depth == _numberOfChannel, @"The input tensor's depth is not identical to the initialized.");
    NSAssert(channelIndex < _numberOfChannel, @"Invalid channel index");
    NSAssert(dst.shape->depth == 4, @"MetalTensorSlice only can output one channel each time.");
    NSAssert(src.shape->row == dst.shape->row && src.shape->column == dst.shape->column, @"MetalTensorSlice output tensor should be the same row and column.");
    
    [_dataSource activeChannelAtIndex:channelIndex];
    [_convolution reloadWeightsAndBiasesFromDataSource];
    [_convolution encodeToCommandBuffer:commandBuffer sourceImage:src.content destinationImage:dst.content];
}

@end
