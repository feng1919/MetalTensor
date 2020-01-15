//
//  MTTotalVariationLayer.m
//  MetalTensor
//
//  Created by Feng Stone on 2020/1/15.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import "MTTotalVariationLayer.h"
#import "MTChannelReduce.h"
#import "MTTensorCache.h"

@interface TVDataSource : NSObject <MPSCNNConvolutionDataSource, MPSNNPadding> {
    
    MPSCNNConvolutionDescriptor *_convDesc;
    float32_t *_data;
    id<MTLDevice> _device;
}

//  0:  Hirizontal
//  1:  vertical
@property (nonatomic, assign) int direction;

@property (nonatomic, assign) int numberOfChannels;

- (void)compile:(id<MTLDevice>)device;

@end

@implementation TVDataSource

- (instancetype)initWithCoder:(NSCoder *)coder {
    if (self = [super init]) {
        self.direction = [coder decodeIntForKey:@"direction"];
        self.numberOfChannels = [coder decodeIntForKey:@"numberOfChannels"];
        id<MTLDevice> device = [coder decodeObjectForKey:@"device"];
        if (device) {
            [self compile:device];
        }
    }
    return self;
}

- (void)encodeWithCoder:(NSCoder *)coder {
    [coder encodeInt:_direction forKey:@"direction"];
    [coder encodeInt:_numberOfChannels forKey:@"numberOfChannels"];
    [coder encodeObject:_device forKey:@"device"];
}

+ (BOOL)supportsSecureCoding {
    return YES;
}

- (id)copyWithZone:(NSZone *)zone {
    TVDataSource *item = [TVDataSource allocWithZone:zone];
    item.numberOfChannels = _numberOfChannels;
    [item compile:_device];
    return item;
}

- (id)copy {
    TVDataSource *reduce = [[TVDataSource alloc] init];
    reduce.numberOfChannels = _numberOfChannels;
    [reduce compile:_device];
    return reduce;
}

- (void)compile:(id<MTLDevice>)device {
    if (!device) {
        return;
    }
    
    NSAssert(_direction == 0 || _direction == 1, @"Invalid direction.");
    
    _device = device;
    _convDesc = [MPSCNNDepthWiseConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:_direction==0?2:1
                                                                                 kernelHeight:_direction==1?1:2
                                                                         inputFeatureChannels:_numberOfChannels
                                                                        outputFeatureChannels:_numberOfChannels];
    [_convDesc setStrideInPixelsX:1];
    [_convDesc setStrideInPixelsY:1];
    
    MPSNNNeuronDescriptor *powerDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypePower a:1.0f b:0.0f c:2.0f];
    [_convDesc setFusedNeuronDescriptor:powerDesc];
    
    _data = calloc(_numberOfChannels*2, sizeof(float32_t));
    for (int i = 0; i < _numberOfChannels; i++) {
        _data[i*2] = 1.0f;
        _data[i*2+1] = -1.0f;
    }
}

- (void)dealloc {
    free(_data);
}

-(MPSNNPaddingMethod) paddingMethod {
    return MPSNNPaddingMethodAlignCentered | MPSNNPaddingMethodAddRemainderToTopLeft | MPSNNPaddingMethodSizeSame;
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

@interface MTTotalVariationLayer () {
    
    TVDataSource *_dataSourceHorizontal;
    MPSCNNConvolution *_convHorizontal;
    
    TVDataSource *_dataSourceVertical;
    MPSCNNConvolution *_convVertical;
    
    MTChannelReduce *_channelReduce;
    MPSCNNPoolingAverage *_pooling;
    MPSCNNAdd *_add;
}

@end

@implementation MTTotalVariationLayer

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    DataShape *inputShape = &_inputShapes[0];
    
    _dataSourceHorizontal = [[TVDataSource alloc] init];
    _dataSourceHorizontal.direction = 0;
    _dataSourceHorizontal.numberOfChannels = inputShape->depth;
    [_dataSourceHorizontal compile:device];
    
    _dataSourceVertical = [[TVDataSource alloc] init];
    _dataSourceVertical.direction = 1;
    _dataSourceVertical.numberOfChannels = inputShape->depth;
    [_dataSourceVertical compile:device];
    
    _outputShape = DataShapeMake(1, 1, 1);
    
    _convHorizontal = [[MPSCNNConvolution alloc] initWithDevice:device weights:_dataSourceHorizontal];
    [_convHorizontal setEdgeMode:MPSImageEdgeModeClamp];
    
    _convVertical = [[MPSCNNConvolution alloc] initWithDevice:device weights:_dataSourceVertical];
    [_convVertical setEdgeMode:MPSImageEdgeModeClamp];
    
    _add = [[MPSCNNAdd alloc] initWithDevice:device];
    _add.primaryScale = (float)(inputShape->row*inputShape->column);
    _add.secondaryScale = (float)(inputShape->row*inputShape->column);
    
    _channelReduce = [[MTChannelReduce alloc] initWithReduceType:ReduceTypeSum numberOfChannels:inputShape->depth];
    _pooling = [[MPSCNNPoolingAverage alloc] initWithDevice:device
                                                kernelWidth:inputShape->column
                                               kernelHeight:inputShape->row
                                            strideInPixelsX:inputShape->column
                                            strideInPixelsY:inputShape->row];
    _pooling.offset = MPSOffsetMake(inputShape->column>>1, inputShape->row>>1, 0);
}

- (void)setInputShape:(DataShape *)dataShape atIndex:(NSInteger)imageIndex {
    [super setInputShape:dataShape atIndex:imageIndex];
    
    DataShape *inputShape = &_inputShapes[0];
    _pooling = [[MPSCNNPoolingAverage alloc] initWithDevice:_device
                                                kernelWidth:inputShape->column
                                               kernelHeight:inputShape->row
                                            strideInPixelsX:inputShape->column
                                            strideInPixelsY:inputShape->row];
    _pooling.offset = MPSOffsetMake(inputShape->column>>1, inputShape->row>>1, 0);
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    DataShape *inputShape = &_inputShapes[0];
    MetalTensor sourceTensor = _inputImages[@(0)];
    MetalTensor convResult = [[MTTensorCache sharedCache] fetchTensorWithShape1:DataShapeMake(inputShape->row, inputShape->column, inputShape->depth)
                                                                            commandBuffer:commandBuffer];
    
    MetalTensor poolingResult = [[MTTensorCache sharedCache] fetchTensorWithShape1:DataShapeMake(1, 1, inputShape->depth) commandBuffer:commandBuffer];
    MetalTensor horizontalResult = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape commandBuffer:commandBuffer];
    MetalTensor verticalResult = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape commandBuffer:commandBuffer];
    
    //  Computing horizontal variation.
    [_convHorizontal encodeToCommandBuffer:commandBuffer
                               sourceImage:sourceTensor.content
                          destinationImage:convResult.content];
    [_pooling encodeToCommandBuffer:commandBuffer
                        sourceImage:convResult.content
                   destinationImage:poolingResult.content];
    [_channelReduce reduceOnCommandBuffer:commandBuffer
                             sourceTensor:poolingResult
                        destinationTensor:horizontalResult];
    
    //  Computing vertical variation.
    [_convVertical encodeToCommandBuffer:commandBuffer
                             sourceImage:sourceTensor.content
                        destinationImage:convResult.content];
    [_pooling encodeToCommandBuffer:commandBuffer
                        sourceImage:convResult.content
                   destinationImage:poolingResult.content];
    [_channelReduce reduceOnCommandBuffer:commandBuffer
                             sourceTensor:poolingResult
                        destinationTensor:verticalResult];
    
    //  Sum the vertical result and the horizontal result.
    //  We use pooling average to sum spatial variation, so we have to scale the widthxheight.
    _image = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape commandBuffer:commandBuffer];
    _image.source = self;
    [_add encodeToCommandBuffer:commandBuffer
                   primaryImage:horizontalResult.content
                 secondaryImage:verticalResult.content
               destinationImage:_image.content];
    
    [verticalResult unlock];
    [horizontalResult unlock];
    [poolingResult unlock];
    [convResult unlock];
    
    if (!_needBackward) {
        [self removeCachedImages];
    }
}

@end
