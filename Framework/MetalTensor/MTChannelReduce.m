//
//  MTChannelReduce.m
//  MetalTensor
//
//  Created by Feng Stone on 2020/1/14.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import "MTChannelReduce.h"
#import "MTTensorCache.h"

@interface ChannelReduceDataSource : NSObject <MPSCNNConvolutionDataSource> {
    
    MPSCNNConvolutionDescriptor *_convDesc;
    float32_t *_data;
    id<MTLDevice> _device;
}

@property (nonatomic, assign) int numberOfChannels;
@property (nonatomic, assign) float weight;

- (void)compile:(id<MTLDevice>)device;

@end

@implementation ChannelReduceDataSource

- (id)copyWithZone:(NSZone *)zone {
    ChannelReduceDataSource *item = [ChannelReduceDataSource allocWithZone:zone];
    item.numberOfChannels = _numberOfChannels;
    item.weight = _weight;
    [item compile:_device];
    return item;
}

- (id)copy {
    ChannelReduceDataSource *reduce = [[ChannelReduceDataSource alloc] init];
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
    _convDesc = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:1
                                                                        kernelHeight:1
                                                                inputFeatureChannels:_numberOfChannels
                                                               outputFeatureChannels:1];
    [_convDesc setStrideInPixelsX:1];
    [_convDesc setStrideInPixelsY:1];
    
    int numberOfChannels = (_numberOfChannels+3)>>2<<2;
    _data = calloc(numberOfChannels, sizeof(float32_t));
    for (int i = 0; i < numberOfChannels; i++) {
        _data[i] = _weight;
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


@implementation MTChannelReduce {
    ReduceType _type;
    int _numberOfChannels;
    ChannelReduceDataSource *_dataSource;
    id<MTLDevice> _device;
    MPSCNNConvolution *_convolution;
}

- (instancetype)initWithReduceType:(ReduceType)type numberOfChannels:(int)numberOfChannels {
    if (self = [super init]) {
        NSAssert(type == ReduceTypeSum || type == ReduceTypeMean,
                 @"The reduce type is not supported, currently sum and mean operations are valid.");
        _type = type;
        _numberOfChannels = numberOfChannels;
    }
    return self;
}

- (void)compile:(id<MTLDevice>)device {
    _device = device;
    _dataSource = [[ChannelReduceDataSource alloc] init];
    _dataSource.numberOfChannels = _numberOfChannels;
    _dataSource.weight = _type==ReduceTypeSum?1.0f:1.0f/(float)_numberOfChannels;
    [_dataSource compile:device];
    
    _convolution = [[MPSCNNConvolution alloc] initWithDevice:device weights:_dataSource];
}

- (void)reduceOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer sourceTensor:(MetalTensor)src destinationTensor:(MetalTensor)dst {
    [_convolution encodeToCommandBuffer:commandBuffer sourceImage:src.content destinationImage:dst.content];
}

@end
