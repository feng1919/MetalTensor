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
#import "MTImageTensor.h"
#import "MTChannelCompress.h"

@implementation MIGramMatrixLayer {
@private
    MetalTensorSlice *_slice;
    MPSCNNMultiply *_multiply;
    MPSCNNPoolingAverage *_mean;
    MPSNNReduceFeatureChannelsSum *_reduceChannels;
    MPSCNNNeuron *_neuron;
    MTChannelCompress *_compress;
    
    DataShape _oneChannelShape;
    DataShape _multiplyShape;
}

#pragma mark - override

- (void)initialize {
    _weight = 1.0f;
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
    
    if (_needBackward) {
        _reduceChannels = [[MPSNNReduceFeatureChannelsSum alloc] initWithDevice:device];
        
//        _reduceChannels.weight = 2.0f/(float)(inputShape->column*inputShape->row)/_weight/_weight;    //  It's MPS's bug here, not work on iOS 13.3.
        
        MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeLinear a:2.0f/(float)(inputShape->column*inputShape->row)/_weight/_weight b:0.0f c:0.0f];
        _neuron = [[MPSCNNNeuron alloc] initWithDevice:device neuronDescriptor:neuronDesc];
        
        _compress = [[MTChannelCompress alloc] initWithNumberOfChannels:inputShape->depth*4];
        _compress.alpha = 1.0f;
        [_compress compile:device];
    }
}

- (void)updateOutputShape {
    if (_device) {
        
        DataShape *inputShape = &_inputShapes[0];
        _outputShape = DataShapeMake(1, inputShape->depth, inputShape->depth);
        _oneChannelShape = DataShapeMake(inputShape->row, inputShape->column, 4);
        _multiplyShape = *inputShape;
    }
}

#pragma mark - MTTensorForward Delegate

- (void)setInputShape:(DataShape *)dataShape atIndex:(NSInteger)imageIndex {
    [super setInputShape:dataShape atIndex:imageIndex];
    
    if (_device) {

        DataShape *inputShape = &_inputShapes[0];
        _mean = [[MPSCNNPoolingAverage alloc] initWithDevice:_device
                                                 kernelWidth:inputShape->column
                                                kernelHeight:inputShape->row
                                             strideInPixelsX:inputShape->column
                                             strideInPixelsY:inputShape->row];
        _mean.offset = MPSOffsetMake(inputShape->column>>1, inputShape->row>>1, 0);
        
//        _reduceChannels.weight = 2.0f/(float)(inputShape->column*inputShape->row)/_weight/_weight;    //  It's MPS's bug here, not work on iOS 13.3.
        
        MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeLinear a:2.0f/(float)(inputShape->column*inputShape->row)/_weight/_weight b:0.0f c:0.0f];
        _neuron = [[MPSCNNNeuron alloc] initWithDevice:_device neuronDescriptor:neuronDesc];
    }
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);

    MetalTensor sourceTensor = _inputImages[@(0)];
    DataShape *inputShape = &_inputShapes[0];
    MetalTensor oneChannel = [[MTTensorCache sharedCache] fetchTensorWithShape:&_oneChannelShape commandBuffer:commandBuffer];
    MetalTensor multiplyImage = [[MTTensorCache sharedCache] fetchTensorWithShape:&_multiplyShape commandBuffer:commandBuffer];
    
    _image = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape commandBuffer:commandBuffer];
    _image.source = self;

    _multiply.secondaryStrideInPixelsX = 1;
    _multiply.secondaryStrideInPixelsY = 1;
    _multiply.secondaryStrideInFeatureChannels = 0;
    
    _multiply.primaryScale = _weight;
    _multiply.secondaryScale = _weight;
    
    MTLRegion clipRect;
    clipRect.origin = MTLOriginMake(0, 0, 0);
    clipRect.size = MTLSizeMake(1, 1, -1);
    
    for (int i = 0; i < inputShape->depth; i++) {
        [_slice sliceTensor:sourceTensor
                   toTensor:oneChannel
               channelIndex:i
              commandBuffer:commandBuffer];
        [_multiply encodeToCommandBuffer:commandBuffer
                            primaryImage:sourceTensor.content
                          secondaryImage:oneChannel.content
                        destinationImage:multiplyImage.content];
        clipRect.origin.x = i;
        [_mean setClipRect:clipRect];
        [_mean encodeToCommandBuffer:commandBuffer
                         sourceImage:multiplyImage.content
                    destinationImage:_image.content];
    }
    
    [oneChannel unlock];
    [multiplyImage unlock];
    
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
    BackwardTarget backwardTarget = sourceTensor.source;
    NSAssert(backwardTarget, @"Invalid backward target...");
    
    int numOfChannels = sourceTensor.shape->depth;
    MetalTensor gradients0 = [[MTTensorCache sharedCache] fetchTensorWithShape:sourceTensor.shape commandBuffer:commandBuffer];
    MetalTensor gradients1 = [[MTTensorCache sharedCache] fetchTensorWithShape1:DataShapeMake(sourceTensor.shape->row, sourceTensor.shape->column, 1)
                                                                  commandBuffer:commandBuffer];
    MetalTensor result = [[MTTensorCache sharedCache] fetchTensorWithShape1:DataShapeMake(sourceTensor.shape->row, sourceTensor.shape->column, 4*numOfChannels)
                                                              commandBuffer:commandBuffer];

    //  Make a channel-wise multiplication.
    _multiply.secondaryStrideInPixelsX = 0;
    _multiply.secondaryStrideInPixelsY = 0;
    _multiply.secondaryStrideInFeatureChannels = 1;
    _multiply.secondaryOffset = MPSOffsetMake(0, 0, 0);
    _multiply.primaryScale = 1.0f;
    _multiply.secondaryScale = 1.0f;
    
    for (int i = 0; i < numOfChannels; i++) {
        //  Data of each channel multipy its respect weight.
        [_multiply setSecondaryOffset:MPSOffsetMake(i, 0, 0)];
        [_multiply encodeToCommandBuffer:commandBuffer
                            primaryImage:sourceTensor.content
                          secondaryImage:_gradient.content
                        destinationImage:gradients0.content];
    
        //  Reduce sum the feature channels.
        [_reduceChannels encodeToCommandBuffer:commandBuffer
                                   sourceImage:gradients0.content
                              destinationImage:gradients1.content];
    
        [_neuron setDestinationFeatureChannelOffset:i<<2];
        [_neuron encodeToCommandBuffer:commandBuffer
                           sourceImage:gradients1.content
                      destinationImage:result.content];
    }
    [gradients1 unlock];
    
    [_compress compressOnCommandBuffer:commandBuffer sourceTensor:result destinationTensor:gradients0];
    [result unlock];
    
    [self removeGradient];
    [self removeCachedImages];
    
    [backwardTarget setGradient:gradients0 forwardTarget:self];
    [gradients0 unlock];
    
    [backwardTarget gradientReadyOnCommandBuffer:commandBuffer forwardTarget:self];
    
}

@end
