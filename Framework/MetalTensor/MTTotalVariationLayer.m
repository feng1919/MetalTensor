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
#import "MTImageTensor.h"

@interface TVDataSource : NSObject <MPSCNNConvolutionDataSource, MPSNNPadding> {
    
    MPSCNNConvolutionDescriptor *_convDesc;
    float32_t *_data;
    id<MTLDevice> _device;
}

//  0:  Hirizontal
//  1:  vertical
@property (nonatomic, assign) int direction;

@property (nonatomic, assign) int numberOfChannels;
@property (nonatomic, assign) float scale;

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
        self.scale = [coder decodeIntForKey:@"scale"];
    }
    return self;
}

- (void)encodeWithCoder:(NSCoder *)coder {
    [coder encodeInt:_direction forKey:@"direction"];
    [coder encodeInt:_numberOfChannels forKey:@"numberOfChannels"];
    [coder encodeObject:_device forKey:@"device"];
    [coder encodeFloat:_scale forKey:@"scale"];
}

+ (BOOL)supportsSecureCoding {
    return YES;
}

- (id)copyWithZone:(NSZone *)zone {
    TVDataSource *item = [TVDataSource allocWithZone:zone];
    item.numberOfChannels = _numberOfChannels;
    item.scale = _scale;
    [item compile:_device];
    return item;
}

- (id)copy {
    TVDataSource *reduce = [[TVDataSource alloc] init];
    reduce.numberOfChannels = _numberOfChannels;
    reduce.scale = _scale;
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
                                                                                 kernelHeight:_direction==0?1:2
                                                                         inputFeatureChannels:_numberOfChannels
                                                                        outputFeatureChannels:_numberOfChannels];
    [_convDesc setStrideInPixelsX:1];
    [_convDesc setStrideInPixelsY:1];
    
    MPSNNNeuronDescriptor *powerDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypePower a:1.0f b:0.0f c:2.0f];
    [_convDesc setFusedNeuronDescriptor:powerDesc];
    
    _data = calloc(_numberOfChannels*2, sizeof(float32_t));
    for (int i = 0; i < _numberOfChannels; i++) {
        _data[i*2] = _scale;
        _data[i*2+1] = -_scale;
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
    
    MPSCNNNeuron *_neuron;
    MPSCNNSubtract *_subtract;
    MPSCNNNeuron *_neuronScale;
}

@end

@implementation MTTotalVariationLayer

- (void)initialize {
    _alpha = 1.0f;
    _scale = 255.0f;
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    DataShape *inputShape = &_inputShapes[0];
    
    _dataSourceHorizontal = [[TVDataSource alloc] init];
    _dataSourceHorizontal.direction = 0;
    _dataSourceHorizontal.scale = _scale;
    _dataSourceHorizontal.numberOfChannels = inputShape->depth;
    [_dataSourceHorizontal compile:device];
    
    _dataSourceVertical = [[TVDataSource alloc] init];
    _dataSourceVertical.direction = 1;
    _dataSourceVertical.scale = _scale;
    _dataSourceVertical.numberOfChannels = inputShape->depth;
    [_dataSourceVertical compile:device];
    
    _convHorizontal = [[MPSCNNConvolution alloc] initWithDevice:device weights:_dataSourceHorizontal];
    [_convHorizontal setOffset:MPSOffsetMake(1, 0, 0)];
    [_convHorizontal setEdgeMode:MPSImageEdgeModeClamp];
//    clipRect.origin.x = 1;
//    clipRect.origin.y = 0;
//    [_convHorizontal setClipRect:clipRect];
    
    _convVertical = [[MPSCNNConvolution alloc] initWithDevice:device weights:_dataSourceVertical];
    [_convVertical setOffset:MPSOffsetMake(0, 1, 0)];
    [_convVertical setEdgeMode:MPSImageEdgeModeClamp];
//    clipRect.origin.x = 0;
//    clipRect.origin.y = 1;
//    [_convVertical setClipRect:clipRect];
    
    _channelReduce = [[MTChannelReduce alloc] initWithReduceType:ReduceTypeSum numberOfChannels:(inputShape->depth*2+3)>>2<<2];
    _channelReduce.alpha = _alpha/(float)inputShape->depth/2.0f;
    [_channelReduce compile:device];
    
    int pooling_row = inputShape->row-1;
    int pooling_column = inputShape->column-1;
    _pooling = [[MPSCNNPoolingAverage alloc] initWithDevice:_device
                                                kernelWidth:pooling_column
                                               kernelHeight:pooling_row
                                            strideInPixelsX:pooling_column
                                            strideInPixelsY:pooling_row];
    _pooling.offset = MPSOffsetMake(pooling_column>>1, pooling_row>>1, 0);
    
    if (_needBackward) {
        MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeLinear a:4.0f b:0.0f c:0.0f];
        _neuron = [[MPSCNNNeuron alloc] initWithDevice:device neuronDescriptor:neuronDesc];
        
        _subtract = [[MPSCNNSubtract alloc] initWithDevice:device];
        [_subtract setSecondaryEdgeMode:MPSImageEdgeModeClamp];
        
        neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeLinear
                                                                      a:_alpha/(float)Product(inputShape)
                                                                      b:0.0f
                                                                      c:0.0f];
        _neuronScale = [[MPSCNNNeuron alloc] initWithDevice:device neuronDescriptor:neuronDesc];
    }
}

- (void)updateOutputShape {
    _outputShape = DataShapeMake(1, 1, 1);
}

- (void)setInputShape:(DataShape *)dataShape atIndex:(NSInteger)imageIndex {
    [super setInputShape:dataShape atIndex:imageIndex];
    
    DataShape *inputShape = &_inputShapes[0];
    int pooling_row = inputShape->row-1;
    int pooling_column = inputShape->column-1;
    _pooling = [[MPSCNNPoolingAverage alloc] initWithDevice:_device
                                                kernelWidth:pooling_column
                                               kernelHeight:pooling_row
                                            strideInPixelsX:pooling_column
                                            strideInPixelsY:pooling_row];
    _pooling.offset = MPSOffsetMake(pooling_column>>1, pooling_row>>1, 0);

    if (_needBackward) {
        MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeLinear
                                                                                             a:_alpha/(float)Product(inputShape)
                                                                                             b:0.0f
                                                                                             c:0.0f];
        _neuronScale = [[MPSCNNNeuron alloc] initWithDevice:_device neuronDescriptor:neuronDesc];
    }
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    DataShape *inputShape = &_inputShapes[0];
    MetalTensor sourceTensor = _inputImages[@(0)];
    int depth = (inputShape->depth+3)>>2<<2;
    MetalTensor convResult = [[MTTensorCache sharedCache] fetchTensorWithShape1:DataShapeMake(inputShape->row-1, inputShape->column-1, depth*2)
                                                                       dataType:_dataType
                                                                  commandBuffer:commandBuffer];
    MetalTensor powerResult = [[MTTensorCache sharedCache] fetchTensorWithShape1:DataShapeMake(inputShape->row-1, inputShape->column-1, depth*2)
                                                                        dataType:_dataType
                                                                   commandBuffer:commandBuffer];
    MetalTensor poolingResult = [[MTTensorCache sharedCache] fetchTensorWithShape1:DataShapeMake(1, 1, depth*2)
                                                                          dataType:_dataType
                                                                     commandBuffer:commandBuffer];
    _image = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape
                                                      dataType:_dataType
                                                 commandBuffer:commandBuffer];
    _image.source = self;
    
    [_convHorizontal encodeToCommandBuffer:commandBuffer
                               sourceImage:sourceTensor.content
                          destinationImage:convResult.content];
    [_convVertical setDestinationFeatureChannelOffset:depth];
    [_convVertical encodeToCommandBuffer:commandBuffer
                             sourceImage:sourceTensor.content
                        destinationImage:convResult.content];
    [_pooling encodeToCommandBuffer:commandBuffer
                        sourceImage:convResult.content
                   destinationImage:poolingResult.content];
    [_channelReduce reduceOnCommandBuffer:commandBuffer
                             sourceTensor:poolingResult
                        destinationTensor:_image];
    
    [powerResult unlock];
    [poolingResult unlock];
    [convResult unlock];
    
    if (!_needBackward) {
        [self removeCachedImages];
    }

#if DEBUG
    if (self.dumpResult) {
        [self saveTensor:_image onCommandBuffer:commandBuffer];
    }
#endif
}

- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    /*
     *  for pixel at (x, y), its derivative is
     *  df/dv = (v(x,y)-v(x-1,y)) + (v(x,y)-v(x+1,y)) + (v(x,y)-v(x,y-1)) + (v(x,y)-v(x,y+1))
     *        = 4.0 * v(x,y) - v(x-1,y) - v(x+1,y) - v(x,y-1) - v(x,y+1)
     *
     */
    
    MetalTensor sourceTensor = _inputImages[@(0)];
    BackwardTarget backwardTarget = sourceTensor.source;
    NSAssert(backwardTarget, @"Invalid backward target...");
    
    DataShape *inputShape = &_inputShapes[0];
    MetalTensor copiedTensor = [[MTTensorCache sharedCache] fetchTensorWithShape:inputShape
                                                                        dataType:_dataType
                                                                   commandBuffer:commandBuffer];
    [_neuron encodeToCommandBuffer:commandBuffer
                       sourceImage:sourceTensor.content
                  destinationImage:copiedTensor.content];
    
    MetalTensor resultTensor = [[MTTensorCache sharedCache] fetchTensorWithShape:inputShape
                                                                        dataType:_dataType
                                                                   commandBuffer:commandBuffer];
    //  right
    [_subtract setSecondaryOffset:MPSOffsetMake(1, 0, 0)];
    [_subtract encodeToCommandBuffer:commandBuffer
                        primaryImage:copiedTensor.content
                      secondaryImage:sourceTensor.content
                    destinationImage:resultTensor.content];
    //  left
    [_subtract setSecondaryOffset:MPSOffsetMake(-1, 0, 0)];
    [_subtract encodeToCommandBuffer:commandBuffer
                        primaryImage:resultTensor.content
                      secondaryImage:sourceTensor.content
                    destinationImage:copiedTensor.content];
    //  down
    [_subtract setSecondaryOffset:MPSOffsetMake(0, 1, 0)];
    [_subtract encodeToCommandBuffer:commandBuffer
                        primaryImage:copiedTensor.content
                      secondaryImage:sourceTensor.content
                    destinationImage:resultTensor.content];
    //  up
    [_subtract setSecondaryOffset:MPSOffsetMake(0, -1, 0)];
    [_subtract encodeToCommandBuffer:commandBuffer
                        primaryImage:resultTensor.content
                      secondaryImage:sourceTensor.content
                    destinationImage:copiedTensor.content];
    
    //  scale
    [_neuronScale encodeToCommandBuffer:commandBuffer
                            sourceImage:copiedTensor.content
                       destinationImage:resultTensor.content];
    
    [copiedTensor unlock];
    [self removeCachedImages];
    [self removeGradient];
    
    if (self.stopGradient) {
        [self.blit encodeToCommandBuffer:commandBuffer
                             sourceImage:resultTensor.content
                        destinationImage:self.savedGradients.content];
        [resultTensor unlock];
    }
    else {
        [backwardTarget setGradient:resultTensor forwardTarget:self];
        [resultTensor unlock];
        [backwardTarget gradientReadyOnCommandBuffer:commandBuffer forwardTarget:self];
    }
}

@end
