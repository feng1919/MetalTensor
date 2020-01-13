//
//  MetalTensorSpacialReduce.m
//  MetalTensor
//
//  Created by Feng Stone on 2020/1/13.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import "MetalTensorSpacialReduce.h"
#import "MTTensorCache.h"
#import "numpy.h"

@interface ReduceDataSource : NSObject <MPSCNNConvolutionDataSource> {
    
    DataShape _dataShape;
    MPSCNNConvolutionDescriptor *_convDesc;
    float32_t *_data;
    id<MTLDevice> _device;
}

@property (nonatomic, assign) int kernelWidth;
@property (nonatomic, assign) int kernelHeight;
@property (nonatomic, assign) float weight;
@property (nonatomic, readonly) MPSOffset offset;

- (void)compile:(id<MTLDevice>)device;

@end

@implementation ReduceDataSource

- (id)copyWithZone:(NSZone *)zone {
    ReduceDataSource *item = [ReduceDataSource allocWithZone:zone];
    item->_dataShape = _dataShape;
    item.kernelWidth = _kernelWidth;
    item.kernelHeight = _kernelHeight;
    [item compile:_device];
    return item;
}

- (id)copy {
    ReduceDataSource *reduce = [[ReduceDataSource alloc] initWithInputShape:&_dataShape];
    reduce.kernelWidth = _kernelWidth;
    reduce.kernelHeight = _kernelHeight;
    [reduce compile:_device];
    return reduce;
}

- (instancetype)initWithInputShape:(DataShape *)dataShape {
    if (self = [super init]) {
        _dataShape = *dataShape;
    }
    return self;
}

- (void)compile:(id<MTLDevice>)device {
    if (!device) {
        return;
    }
    _device = device;
    _convDesc = [MPSCNNDepthWiseConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:_kernelWidth
                                                                                 kernelHeight:_kernelHeight
                                                                         inputFeatureChannels:_dataShape.depth
                                                                        outputFeatureChannels:_dataShape.depth];
    [_convDesc setStrideInPixelsX:_kernelWidth];
    [_convDesc setStrideInPixelsY:_kernelHeight];
    
    _offset.x = _kernelWidth >> 1;
    _offset.y = _kernelHeight >> 1;
    _offset.z = 0;
    
    _data = calloc(_kernelHeight*_kernelWidth*_dataShape.depth, sizeof(float32_t));
    for (int i = 0; i < _kernelHeight*_kernelWidth*_dataShape.depth; i++) {
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
    return [NSString stringWithFormat:@"ReduceDataSource: %@", NSStringFromDataShape(&_dataShape)];
}

- (MPSCNNConvolutionDescriptor *)descriptor {
    return _convDesc;
}

@end


@implementation MetalTensorSpacialReduce {
    id<MTLDevice> _device;
    MPSCNNConvolution *_convolution;
    ReduceDataSource *_dataSource;
    BOOL _twice;
}

- (void)compile:(id<MTLDevice>)device {
    NSAssert(DataShapeValid(&_inputShape), @"The input shape is not initialized.");
    NSAssert(_axis > 0, @"Invalid reduce axis.");
    NSAssert(_type == ReduceTypeMean || _type == ReduceTypeSum, @"Invalid reduce type.");
    
    _device = device;
    _clipRect = MPSRectNoClip;
    
    int kernelWidth = (_axis & ReduceAxisColumn) == 0?1: _inputShape.column;
    int kernelHeight = (_axis & ReduceAxisRow) == 0?1:_inputShape.row;
    float weight = 1.0f/ (float)(kernelHeight*kernelWidth);
    
    if (kernelWidth > 32 || kernelHeight > 32) {
        // The dimensions are too large, separate to two steps.
        kernelWidth = ceilf(sqrtf((float)_inputShape.column));
        kernelHeight = ceilf(sqrtf((float)_inputShape.row));
        weight = sqrtf(weight);
        _twice = YES;
    }
    
    _dataSource = [[ReduceDataSource alloc] initWithInputShape:&_inputShape];
    _dataSource.kernelWidth = kernelWidth;
    _dataSource.kernelHeight = kernelHeight;
    _dataSource.weight = weight;
    [_dataSource compile:device];
    
    _convolution = [[MPSCNNConvolution alloc] initWithDevice:device weights:_dataSource];
    _convolution.offset = _dataSource.offset;
//    _convolution.options = MPSKernelOptionsDisableInternalTiling;
}

- (void)reduceOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                 sourceTensor:(MetalTensor)sourceTensor
            destinationTensor:(MetalTensor)destinationTensor {
    if (_twice) {
        DataShape tempShape = DataShapeMake(_dataSource.kernelHeight, _dataSource.kernelWidth, _inputShape.depth);
        MetalTensor temp = [[MTTensorCache sharedCache] fetchTensorWithShape:&tempShape source:nil commandBuffer:commandBuffer];
        [temp newContentOnCommandBuffer:commandBuffer];
        [_convolution setClipRect:MPSRectNoClip];
        [_convolution encodeToCommandBuffer:commandBuffer sourceImage:sourceTensor.content destinationImage:temp.content];
        [_convolution setClipRect:_clipRect];
        [_convolution encodeToCommandBuffer:commandBuffer sourceImage:temp.content destinationImage:destinationTensor.content];
        [temp unlock];
    }
    else {
        [_convolution setClipRect:_clipRect];
        [_convolution encodeToCommandBuffer:commandBuffer
                                sourceImage:sourceTensor.content
                           destinationImage:destinationTensor.content];
    }
}

@end
