//
//  MIGramMatrixLayer.m
//  MetalTensor
//
//  Created by Feng Stone on 2019/12/31.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIGramMatrixLayer.h"
#import "MTTensorCache.h"

@implementation MIGramMatrixLayer {
@private
    MPSCNNNeuron *_neuron;
    MPSCNNMultiply *_multiply;
    MPSNNReduceRowMean *_reduceMeanRow;
    MPSNNReduceColumnMean *_reduceMeanColumn;
    MPSNNReduceFeatureChannelsSum *_reduceSumChannels;
    
    DataShape _outputArithmetic;
    DataShape _outputReduceRow;
    DataShape _outputReduceColumn;
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    NSParameterAssert(device);
    
    _outputShape = DataShapeMake(_inputShapes[0].depth, _inputShapes[0].depth, 1);
    _outputArithmetic = _inputShapes[0];
    _outputReduceRow = DataShapeMake(1, _inputShapes[0].column, _inputShapes[0].depth);
    
    MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeNone];
    _neuron = [[MPSCNNNeuron alloc] initWithDevice:device neuronDescriptor:neuronDesc];
    
    _multiply = [[MPSCNNMultiply alloc] initWithDevice:device];
    _multiply.primaryScale = 1.0f;
    _multiply.bias = 0.0f;
    _multiply.primaryStrideInPixelsX = 1;
    _multiply.primaryStrideInPixelsY = 1;
    _multiply.primaryStrideInFeatureChannels = 1;
    _multiply.secondaryScale = 1.0f;
    _multiply.secondaryStrideInPixelsX = 1;
    _multiply.secondaryStrideInPixelsY = 1;
    _multiply.secondaryStrideInFeatureChannels = 0;
    _multiply.destinationFeatureChannelOffset = 0;
    
    _reduceMeanRow = [[MPSNNReduceRowMean alloc] initWithDevice:device];
    _reduceMeanColumn = [[MPSNNReduceColumnMean alloc] initWithDevice:device];
    
    if (_needBackward) {
        _reduceSumChannels = [[MPSNNReduceFeatureChannelsSum alloc] initWithDevice:device];
    }
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);
    
    MetalTensor sourceTensor = _inputImages[@(0)];
    int numOfChannels = _inputShapes[0].depth;
    MetalTensor secondaryImage = [[MTTensorCache sharedCache] fetchTensorWithShape:&_inputShapes[0] source:self commandBuffer:commandBuffer];
    [secondaryImage newContentOnCommandBuffer:commandBuffer];
    MetalTensor arithmeticImage = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputArithmetic source:self commandBuffer:commandBuffer];
    [arithmeticImage newContentOnCommandBuffer:commandBuffer];
    MetalTensor reduceRowImage = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputReduceRow source:self commandBuffer:commandBuffer];
    [reduceRowImage newContentOnCommandBuffer:commandBuffer];
    _image = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape source:self commandBuffer:commandBuffer];
    [_image newContentOnCommandBuffer:commandBuffer];
    
    _neuron.offset = MPSOffsetMake(0, 0, 0);
    [_neuron encodeToCommandBuffer:commandBuffer sourceImage:sourceTensor.content destinationImage:secondaryImage.content];
    
    MTLRegion clipRect;
    clipRect.origin = MTLOriginMake(0, 0, 0);
    clipRect.size = MTLSizeMake(_inputShapes[0].column, _inputShapes[0].row, _inputShapes[0].depth);

    _multiply.secondaryStrideInPixelsX = 1;
    _multiply.secondaryStrideInPixelsY = 1;
    _multiply.secondaryStrideInFeatureChannels = 0;
    
    for (int i = 0; i < numOfChannels; i++) {
        [_multiply setSecondaryOffset:MPSOffsetMake(0, 0, i)];
        [_multiply encodeToCommandBuffer:commandBuffer
                              primaryImage:sourceTensor.content
                            secondaryImage:secondaryImage.content
                          destinationImage:arithmeticImage.content];
        
        [_reduceMeanRow encodeToCommandBuffer:commandBuffer
                              sourceImage:arithmeticImage.content
                         destinationImage:reduceRowImage.content];
        
        clipRect.origin.y = i;
        [_reduceMeanColumn setClipRect:clipRect];
        [_reduceMeanColumn encodeToCommandBuffer:commandBuffer
                                 sourceImage:reduceRowImage.content
                            destinationImage:_image.content];
    }
    
    if (!_needBackward) {
        [self removeCachedImages];
    }
    
    [arithmeticImage unlock];
    [reduceRowImage unlock];
    [secondaryImage unlock];
}

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
    MetalTensor gradients0 = [[MTTensorCache sharedCache] fetchTensorWithShape:sourceTensor.shape source:nil commandBuffer:commandBuffer];
    [gradients0 newContentOnCommandBuffer:commandBuffer];
    MetalTensor gradients1 = [[MTTensorCache sharedCache] fetchTensorWithShape:sourceTensor.shape source:nil commandBuffer:commandBuffer];
    [gradients1 newContentOnCommandBuffer:commandBuffer];
    MetalTensor channelsWeights = [[MTTensorCache sharedCache] fetchTensorWithShape:&channelsWeightsShape source:nil commandBuffer:commandBuffer];
    [channelsWeights newContentOnCommandBuffer:commandBuffer];

    //  Make a channel-wise multiplication.
    _multiply.secondaryStrideInPixelsX = 0;
    _multiply.secondaryStrideInPixelsY = 0;
    _multiply.secondaryStrideInFeatureChannels = 1;
    _multiply.secondaryOffset = MPSOffsetMake(0, 0, 0);
    
    for (int i = 0; i < numOfChannels; i++) {
        //  Copy the i-th row gradient.
        _neuron.offset = MPSOffsetMake(0, i, 0);
        [_neuron encodeToCommandBuffer:commandBuffer
                           sourceImage:_gradient.content
                      destinationImage:channelsWeights.content];
        
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
    [sourceTensor.source gradientReadyFromForwardTarget:self onCommandBuffer:commandBuffer];
}

@end
