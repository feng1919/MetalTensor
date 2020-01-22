//
//  MTChannelCompress.m
//  MetalTensor
//
//  Created by Feng Stone on 2020/1/21.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import "MTChannelCompress.h"
#import "MTTensorCache.h"

@interface ChannelCompressDataSource : NSObject <MPSCNNConvolutionDataSource> {
    
    MPSCNNConvolutionDescriptor *_convDesc;
    float32_t *_data;
    id<MTLDevice> _device;
}

@property (nonatomic, assign) int numberOfChannels;
@property (nonatomic, assign) float weight;

- (void)compile:(id<MTLDevice>)device;

@end

@implementation ChannelCompressDataSource

- (id)copyWithZone:(NSZone *)zone {
    ChannelCompressDataSource *item = [ChannelCompressDataSource allocWithZone:zone];
    item.numberOfChannels = _numberOfChannels;
    item.weight = _weight;
    [item compile:_device];
    return item;
}

- (id)copy {
    ChannelCompressDataSource *reduce = [[ChannelCompressDataSource alloc] init];
    reduce.numberOfChannels = _numberOfChannels;
    reduce.weight = _weight;
    [reduce compile:_device];
    return reduce;
}

- (void)compile:(id<MTLDevice>)device {
    if (!device) {
        return;
    }
    _device = device;
    
    int outputChannels = (_numberOfChannels+3)>>2;
    _convDesc = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:1
                                                                        kernelHeight:1
                                                                inputFeatureChannels:_numberOfChannels
                                                               outputFeatureChannels:outputChannels];
    [_convDesc setStrideInPixelsX:1];
    [_convDesc setStrideInPixelsY:1];
    
    _data = calloc(_numberOfChannels*outputChannels, sizeof(float32_t));
    for (int i = 0; i < outputChannels; i++) {
        int index = i * _numberOfChannels;
        for (int j = 0; j < _numberOfChannels; j++) {
            _data[index+j] = (i*4) == j?_weight:0.0f;
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
    return [NSString stringWithFormat:@"ChannelReduceDataSource: %d", _numberOfChannels];
}

- (MPSCNNConvolutionDescriptor *)descriptor {
    return _convDesc;
}

@end


@implementation MTChannelCompress {
    int _numberOfChannels;
    ChannelCompressDataSource *_dataSource;
    id<MTLDevice> _device;
    MPSCNNConvolution *_convolution;
}

- (instancetype)initWithNumberOfChannels:(int)numberOfChannels {
    if (self = [super init]) {
        _numberOfChannels = numberOfChannels;
        _alpha = 1.0f;
    }
    return self;
}

- (void)compile:(id<MTLDevice>)device {
    _device = device;
    _dataSource = [[ChannelCompressDataSource alloc] init];
    _dataSource.numberOfChannels = _numberOfChannels;
    [_dataSource setWeight:_alpha];
    [_dataSource compile:device];
    
    _convolution = [[MPSCNNConvolution alloc] initWithDevice:device weights:_dataSource];
}

- (void)compressOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer sourceTensor:(MetalTensor)src destinationTensor:(MetalTensor)dst {
    [_convolution encodeToCommandBuffer:commandBuffer
                            sourceImage:src.content
                       destinationImage:dst.content];
}

@end
