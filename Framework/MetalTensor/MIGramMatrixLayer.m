//
//  MIGramMatrixLayer.m
//  MetalTensor
//
//  Created by Feng Stone on 2019/12/31.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIGramMatrixLayer.h"
#import "MTTensorCache.h"
#import "MetalTensorSlice.h"

@implementation MIGramMatrixLayer {
@private
    MetalTensorSlice *_slice;
    MPSCNNMultiply *_multiply;
    MPSCNNPoolingAverage *_mean;
    MPSNNReduceFeatureChannelsSum *_reduceSumChannels;
    MPSCNNNeuron *_neuron;
    MPSNNReshape *_reshape;
    
    DataShape _oneChannelShape;
    DataShape _multiplyShape;
    DataShape _poolingShape;
    DataShape _reshapeShape;
}

#pragma mark - override

- (void)initialize {
    NSAssert((_inputShapes[0].column&0x01)==0 && (_inputShapes[0].row&0x01)==0, @"The dimensions have to be multiple of 2.");
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    NSParameterAssert(device);
    
    DataShape *inputShape = &_inputShapes[0];
    
    _multiply = [[MPSCNNMultiply alloc] initWithDevice:device];
    _multiply.primaryScale = 1.0f;
    _multiply.bias = 0.0f;
    _multiply.primaryStrideInPixelsX = 1;
    _multiply.primaryStrideInPixelsY = 1;
    _multiply.primaryStrideInFeatureChannels = 1;
    _multiply.primaryOffset = MPSOffsetMake(0, 0, 0);
    _multiply.secondaryScale = 1.0f;
    _multiply.secondaryStrideInPixelsX = 1;
    _multiply.secondaryStrideInPixelsY = 1;
    _multiply.secondaryStrideInFeatureChannels = 0;
    _multiply.secondaryOffset = MPSOffsetMake(0, 0, 0);
    _multiply.destinationFeatureChannelOffset = 0;
    
    _slice = [[MetalTensorSlice alloc] initWithNumberOfChannel:inputShape->depth];
    [_slice compile:device];
    
    _mean = [[MPSCNNPoolingAverage alloc] initWithDevice:device
                                             kernelWidth:inputShape->column
                                            kernelHeight:inputShape->row
                                         strideInPixelsX:inputShape->column
                                         strideInPixelsY:inputShape->row];
    _mean.offset = MPSOffsetMake(inputShape->column>>1, inputShape->row>>1, 0);
    
    MPSNNNeuronDescriptor *desc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeNone];
    _neuron = [[MPSCNNNeuron alloc] initWithDevice:device neuronDescriptor:desc];
    
    _reshape = [[MPSNNReshape alloc] initWithDevice:device];
    
    if (_needBackward) {
        _reduceSumChannels = [[MPSNNReduceFeatureChannelsSum alloc] initWithDevice:device];
    }
}

- (void)updateOutputShape {
    if (_device) {
        
        DataShape *inputShape = &_inputShapes[0];
//        _outputShape = *inputShape;
        _outputShape = DataShapeMake(1, inputShape->depth, 1);
        _oneChannelShape = DataShapeMake(inputShape->row, inputShape->column, 4);
        _multiplyShape = *inputShape;
        _poolingShape = DataShapeMake(1, 1, inputShape->depth);
    }
}

#pragma mark - MTTensorForward Delegate

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);
    
    MetalTensor sourceTensor = _inputImages[@(0)];
    DataShape *inputShape = &_inputShapes[0];
    MetalTensor oneChannel = [[MTTensorCache sharedCache] fetchTensorWithShape:&_oneChannelShape commandBuffer:commandBuffer];
    MetalTensor multiplyImage = [[MTTensorCache sharedCache] fetchTensorWithShape:&_multiplyShape commandBuffer:commandBuffer];
    MetalTensor poolingImage = [[MTTensorCache sharedCache] fetchTensorWithShape:&_poolingShape commandBuffer:commandBuffer];
    
    _image = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape
                                                    dataFormat:TensorDataFormatFloat16
                                                numberOfImages:inputShape->depth
                                                 commandBuffer:commandBuffer];
    _image.source = self;

    _multiply.secondaryStrideInPixelsX = 1;
    _multiply.secondaryStrideInPixelsY = 1;
    _multiply.secondaryStrideInFeatureChannels = 0;
    
    MTLRegion clipRect;
    clipRect.origin = MTLOriginMake(0, 0, 0);
    clipRect.size = MTLSizeMake(1, 1, 1);
    
    for (int i = 0; i < inputShape->depth; i++) {

        [_slice sliceTensor:sourceTensor
                   toTensor:oneChannel
               channelIndex:i
              commandBuffer:commandBuffer];
        [_multiply encodeToCommandBuffer:commandBuffer
                            primaryImage:sourceTensor.content
                          secondaryImage:oneChannel.content
                        destinationImage:multiplyImage.content];    //  384x512x64
        [_mean encodeToCommandBuffer:commandBuffer
                         sourceImage:multiplyImage.content
                    destinationImage:poolingImage.content];

        clipRect.origin.z = i;
        [_reshape setClipRect:clipRect];
        [_reshape encodeToCommandBuffer:commandBuffer sourceImage:poolingImage.content destinationImage:_image.content];
    }
    
    [oneChannel unlock];
    [multiplyImage unlock];
    [poolingImage unlock];
    
    if (!_needBackward) {
        [self removeCachedImages];
    }
}

#pragma mark - MTTensorBackward Delegate
- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    /*
     *  g = Gradient(i), where i stands for i-th row of backward gradients.
     *
     *  df/dv(j) = sum(v(j)*g(j)),  where j stands for j-th channels of
     *  forward intput tensor.
     *
     *  Note: Gramian matrix is axisymmetric, so it's not necessary to compute
     *  vertical gradient.
     *
     */
    
    MetalTensor sourceTensor = _inputImages[@(0)];
    int numOfChannels = sourceTensor.shape->depth;
    DataShape channelsWeightsShape = DataShapeMake(1, 1, numOfChannels);
    MetalTensor gradients0 = [[MTTensorCache sharedCache] fetchTensorWithShape:sourceTensor.shape commandBuffer:commandBuffer];
    MetalTensor gradients1 = [[MTTensorCache sharedCache] fetchTensorWithShape:sourceTensor.shape commandBuffer:commandBuffer];
    MetalTensor channelsWeights = [[MTTensorCache sharedCache] fetchTensorWithShape:&channelsWeightsShape commandBuffer:commandBuffer];

    //  Make a channel-wise multiplication.
    _multiply.secondaryStrideInPixelsX = 0;
    _multiply.secondaryStrideInPixelsY = 0;
    _multiply.secondaryStrideInFeatureChannels = 1;
    _multiply.secondaryOffset = MPSOffsetMake(0, 0, 0);
    
    for (int i = 0; i < numOfChannels; i++) {
        //  Copy the i-th row gradient.
//        _neuron.offset = MPSOffsetMake(0, i, 0);
//        [_neuron encodeToCommandBuffer:commandBuffer
//                           sourceImage:_gradient.content
//                      destinationImage:channelsWeights.content];
        
        //  Data of each channel multipy its respect weight.
        [_multiply encodeToCommandBuffer:commandBuffer
                              primaryImage:sourceTensor.content
                            secondaryImage:channelsWeights.content
                          destinationImage:gradients0.content];
        
        //  Reduce sum the feature channels.
        [_reduceSumChannels setClipRect:MTLRegionMake3D(0, 0, i, sourceTensor.shape->column, sourceTensor.shape->row, 1)];
        [_reduceSumChannels encodeToCommandBuffer:commandBuffer sourceImage:gradients0.content destinationImage:gradients1.content];
    }
    
    [gradients0 unlock];
    [channelsWeights unlock];
    [self removeGradient];
    [self removeCachedImages];
    
    [sourceTensor.source setGradient:gradients1 forwardTarget:self];
    [gradients1 unlock];
    
    [sourceTensor.source gradientReadyOnCommandBuffer:commandBuffer forwardTarget:self];
}

@end
