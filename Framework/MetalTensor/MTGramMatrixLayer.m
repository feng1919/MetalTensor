//
//  MTGramMatrixLayer.m
//  MetalTensor
//
//  Created by Feng Stone on 2020/2/20.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import "MTGramMatrixLayer.h"
#import "MTTensorCache.h"
#import "MetalTensorSlice.h"
#import "MTChannelCompress.h"

@interface MTGramMatrixLayer() {
    
    MPSImageCopyToMatrix *_imageToMatrix;
    MPSMatrixCopyToImage *_matrixToImage;
    DataShape _transTensorShape;
    
    MPSCNNMultiply *_multiply;
    MPSCNNPoolingAverage *_mean;
    MPSNNReduceFeatureChannelsMean *_reduceChannelsMean;
    
    //  backward propagation
    MPSNNReshape *_reshape;
    MPSNNReduceFeatureChannelsSum *_reduceChannels;
    MPSCNNNeuron *_neuron;
    MTChannelCompress *_compress;
}

@end

@implementation MTGramMatrixLayer

#pragma mark - override

- (void)initialize {
    _weight = 1.0f;
    NSAssert((_inputShapes[0].column&0x01)==0 && (_inputShapes[0].row&0x01)==0, @"The dimensions have to be multiple of 2.");
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    DataShape *inputShape = &_inputShapes[0];
    
    //  Accessing of channels in a MPSImage object MUST be
    //  multiple of 4, but we only need one channel at a time.
    //  There is no transpose operation in MPS framework,
    //  so we use MPSMatrix copying operations to work around.
    //  HWC -> CHW
    _imageToMatrix = [[MPSImageCopyToMatrix alloc] initWithDevice:device dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth];
    _matrixToImage = [[MPSMatrixCopyToImage alloc] initWithDevice:device dataLayout:MPSDataLayoutHeightxWidthxFeatureChannels];
    
    _multiply = [[MPSCNNMultiply alloc] initWithDevice:device];
    _multiply.bias = 0.0f;
    _multiply.primaryScale = 1.0f;
    _multiply.primaryStrideInPixelsX = 1;
    _multiply.primaryStrideInPixelsY = 1;
    _multiply.primaryStrideInFeatureChannels = 1;
    _multiply.primaryOffset = MPSOffsetMake(0, 0, 0);
    _multiply.secondaryScale = 1.0f;
    _multiply.secondaryStrideInPixelsX = 1;
    _multiply.secondaryStrideInPixelsY = 1;
    _multiply.secondaryStrideInFeatureChannels = 1;
    _multiply.secondaryOffset = MPSOffsetMake(0, 0, 0);
    _multiply.destinationFeatureChannelOffset = 0;
    
    //  The tensor is tranposed, new column number equals to the original row number.
    _mean = [[MPSCNNPoolingAverage alloc] initWithDevice:device
                                             kernelWidth:inputShape->row
                                            kernelHeight:1
                                         strideInPixelsX:inputShape->row
                                         strideInPixelsY:1];
    _mean.offset = MPSOffsetMake(inputShape->row>>1, 0, 0);
    
    _reduceChannelsMean = [[MPSNNReduceFeatureChannelsMean alloc] initWithDevice:device];

    if (_needBackward) {
        
        _reshape = [[MPSNNReshape alloc] initWithDevice:device];
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
    
    DataShape *inputShape = &_inputShapes[0];
    _outputShape = DataShapeMake(inputShape->depth, inputShape->depth, 1);
    _transTensorShape = DataShapeMake(inputShape->depth, inputShape->row, inputShape->column);
}

- (void)setInputShape:(DataShape *)dataShape atIndex:(NSInteger)imageIndex {
    [super setInputShape:dataShape atIndex:imageIndex];
    
    if (_device) {

        DataShape *inputShape = &_inputShapes[0];
        _mean = [[MPSCNNPoolingAverage alloc] initWithDevice:_device
                                                 kernelWidth:inputShape->row
                                                kernelHeight:1
                                             strideInPixelsX:inputShape->row
                                             strideInPixelsY:1];
        _mean.offset = MPSOffsetMake(inputShape->row>>1, 0, 0);
        
        if (_needBackward) {
            MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeLinear a:2.0f/(float)(inputShape->column*inputShape->row)/_weight/_weight b:0.0f c:0.0f];
            _neuron = [[MPSCNNNeuron alloc] initWithDevice:_device neuronDescriptor:neuronDesc];
        }
    }
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    
    MetalTensor sourceTensor = _inputImages[@(0)];
    DataShape *inputShape = &_inputShapes[0];
    
    MetalMatrix matrix = [[MTTensorCache sharedCache] fetchMatrixWithRows:1
                                                                  columns:Product(inputShape)
                                                                 dataType:_dataType
                                                            commandBuffer:commandBuffer];
    [_imageToMatrix encodeToCommandBuffer:commandBuffer
                              sourceImage:sourceTensor.content
                        destinationMatrix:matrix.content];
    
    MetalTensor transposedTensor = [[MTTensorCache sharedCache] fetchTensorWithShape:&_transTensorShape
                                                                            dataType:_dataType
                                                                       commandBuffer:commandBuffer];
    [_matrixToImage encodeToCommandBuffer:commandBuffer
                             sourceMatrix:matrix.content
                         destinationImage:transposedTensor.content];
    [matrix unlock];
    
    MetalTensor copiedTensor = [[MTTensorCache sharedCache] fetchTensorWithShape:&_transTensorShape
                                                                        dataType:_dataType
                                                                   commandBuffer:commandBuffer];
    MetalTensor multiplyResult = [[MTTensorCache sharedCache] fetchTensorWithShape:&_transTensorShape
                                                                          dataType:_dataType
                                                                     commandBuffer:commandBuffer];
    
    DataShape columnOneShape = DataShapeMake(inputShape->depth, 1, inputShape->column);
    MetalTensor columnOneTensor = [[MTTensorCache sharedCache] fetchTensorWithShape:&columnOneShape
                                                                           dataType:_dataType
                                                                      commandBuffer:commandBuffer];
    
    _image = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape
                                                      dataType:_dataType
                                                 commandBuffer:commandBuffer];
    _image.source = self;
    
    [self.blit encodeToCommandBuffer:commandBuffer
                         sourceImage:transposedTensor.content
                    destinationImage:copiedTensor.content];
    
    _multiply.primaryScale = _weight;
    _multiply.primaryStrideInPixelsX = 1;
    _multiply.primaryStrideInPixelsY = 1;
    _multiply.primaryStrideInFeatureChannels = 1;
    _multiply.primaryOffset = MPSOffsetMake(0, 0, 0);
    _multiply.secondaryScale = _weight;
    _multiply.secondaryStrideInPixelsX = 1;
    _multiply.secondaryStrideInPixelsY = 0;
    _multiply.secondaryStrideInFeatureChannels = 1;
    _multiply.secondaryOffset = MPSOffsetMake(0, 0, 0);
    _multiply.destinationFeatureChannelOffset = 0;
    
    MTLRegion region = MTLRegionMake3D(0, 0, 0, 1, _transTensorShape.row, 1);
    //  Within transposed tensor, the data of each channals
    //  in original tensor actually store in each row.
    for (int i = 0; i < _transTensorShape.row; i++) {
        //  Gram matrix is axis-symmetric, we don't have to compute
        //  the entire matrix,  this can save a half of time.
//        [_multiply setPrimaryOffset:MPSOffsetMake(0, i, 0)];
        [_multiply setSecondaryOffset:MPSOffsetMake(0, i, 0)];
        [_multiply encodeToCommandBuffer:commandBuffer
                            primaryImage:transposedTensor.content
                          secondaryImage:copiedTensor.content
                        destinationImage:multiplyResult.content];
        [_mean encodeToCommandBuffer:commandBuffer
                         sourceImage:multiplyResult.content
                    destinationImage:columnOneTensor.content];
        region.origin.x = i;
//        region.origin.y = i;
//        region.size.height = _transTensorShape.row - i;
        [_reduceChannelsMean setClipRect:region];
        [_reduceChannelsMean encodeToCommandBuffer:commandBuffer
                                       sourceImage:columnOneTensor.content
                                  destinationImage:_image.content];
    }
    
    [copiedTensor unlock];
    [multiplyResult unlock];
    [columnOneTensor unlock];
    [transposedTensor unlock];
    
    if (!_needBackward) {
        [self removeCachedImages];
    }
    
#if DEBUG
    if (self.dumpResult) {
        [self saveTensor:_image onCommandBuffer:commandBuffer];
    }
#endif
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
    DataShape *inputShape = sourceTensor.shape;
    
    MetalTensor reshapeTensor = [[MTTensorCache sharedCache] fetchTensorWithShape1:DataShapeMake(1, inputShape->depth, inputShape->depth)
                                                                          dataType:_dataType
                                                                     commandBuffer:commandBuffer];
    [_reshape encodeToCommandBuffer:commandBuffer
                        sourceImage:_gradient.content
                   destinationImage:reshapeTensor.content];
    [self removeGradient];
    
    int numOfChannels = inputShape->depth;
    MetalTensor gradients0 = [[MTTensorCache sharedCache] fetchTensorWithShape:inputShape
                                                                      dataType:_dataType
                                                                 commandBuffer:commandBuffer];
    MetalTensor gradients1 = [[MTTensorCache sharedCache] fetchTensorWithShape1:DataShapeMake(inputShape->row, inputShape->column, 1)
                                                                       dataType:_dataType
                                                                  commandBuffer:commandBuffer];
    MetalTensor result = [[MTTensorCache sharedCache] fetchTensorWithShape1:DataShapeMake(inputShape->row, inputShape->column, 4*numOfChannels)
                                                                   dataType:_dataType
                                                              commandBuffer:commandBuffer];

    //  Make a channel-wise multiplication.
    _multiply.primaryScale = 1.0f;
    _multiply.primaryStrideInPixelsX = 1;
    _multiply.primaryStrideInPixelsY = 1;
    _multiply.primaryStrideInFeatureChannels = 1;
    _multiply.primaryOffset = MPSOffsetMake(0, 0, 0);
    _multiply.secondaryScale = 1.0f;
    _multiply.secondaryStrideInPixelsX = 0;
    _multiply.secondaryStrideInPixelsY = 0;
    _multiply.secondaryStrideInFeatureChannels = 1;
    _multiply.secondaryOffset = MPSOffsetMake(0, 0, 0);
    _multiply.destinationFeatureChannelOffset = 0;
    
    for (int i = 0; i < numOfChannels; i++) {
        //  Data of each channel multipy its respect weight.
        [_multiply setSecondaryOffset:MPSOffsetMake(i, 0, 0)];
        [_multiply encodeToCommandBuffer:commandBuffer
                            primaryImage:sourceTensor.content
                          secondaryImage:reshapeTensor.content
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
    [reshapeTensor unlock];
    [self removeCachedImages];
    
    [_compress compressOnCommandBuffer:commandBuffer
                          sourceTensor:result
                     destinationTensor:gradients0];
    [result unlock];
    
    if (self.stopGradient) {
        [self.blit encodeToCommandBuffer:commandBuffer
                             sourceImage:gradients0.content
                        destinationImage:self.savedGradients.content];
        [gradients0 unlock];
    }
    else {
        [backwardTarget setGradient:gradients0 forwardTarget:self];
        [gradients0 unlock];
        [backwardTarget gradientReadyOnCommandBuffer:commandBuffer forwardTarget:self];
    }
}

/*
- (void)processGradientsOnCommandBuffer1:(id<MTLCommandBuffer>)commandBuffer {
    
    *
     *  g = Gradient(i), where i stands for i-th row of backward gradients.
     *
     *  df/dv(j) = sum(v(j)*g(j)),  where j stands for j-th channels of
     *  forward intput tensor.
     *
     *  Note: Gramian matrix is axisymmetric, so it's not necessary to compute
     *  vertical gradient.
     *
     *
    
    MetalTensor sourceTensor = _inputImages[@(0)];
    BackwardTarget backwardTarget = sourceTensor.source;
    NSAssert(backwardTarget, @"Invalid backward target...");
    DataShape *inputShape = sourceTensor.shape;
    
    MPSNNReshape *reshape = [[MPSNNReshape alloc] initWithDevice:_device];
    MetalTensor reshapeTensor = [[MTTensorCache sharedCache] fetchTensorWithShape1:DataShapeMake(1, inputShape->depth, inputShape->depth)
                                                                          dataType:_dataType
                                                                     commandBuffer:commandBuffer];
    [reshape encodeToCommandBuffer:commandBuffer sourceImage:_gradient.content destinationImage:reshapeTensor.content];
    [self saveTensor:reshapeTensor onCommandBuffer:commandBuffer];
    [reshapeTensor unlock];
    
    MetalTensor gradients = [_slice sliceTensor:_gradient channelIndex:0 commandBuffer:commandBuffer];
    
    MetalTensor gradients0 = [[MTTensorCache sharedCache] fetchTensorWithShape:&_transTensorShape dataType:_dataType commandBuffer:commandBuffer];
    MetalTensor gradients1 = [[MTTensorCache sharedCache] fetchTensorWithShape1:DataShapeMake(1, _transTensorShape.column, _transTensorShape.depth)
                                                                       dataType:_dataType
                                                                  commandBuffer:commandBuffer];
    MetalTensor transposedResult = [[MTTensorCache sharedCache] fetchTensorWithShape:&_transTensorShape dataType:_dataType
                                                                       commandBuffer:commandBuffer];

    //  Make a channel-wise multiplication.
    _multiply.primaryScale = 1.0f;
    _multiply.primaryStrideInPixelsX = 1;
    _multiply.primaryStrideInPixelsY = 1;
    _multiply.primaryStrideInFeatureChannels = 1;
    _multiply.primaryOffset = MPSOffsetMake(0, 0, 0);
    _multiply.secondaryScale = 1.0f;
    _multiply.secondaryStrideInPixelsX = 0;
    _multiply.secondaryStrideInPixelsY = 1;
    _multiply.secondaryStrideInFeatureChannels = 0;
    _multiply.secondaryOffset = MPSOffsetMake(0, 0, 0);
    _multiply.destinationFeatureChannelOffset = 0;
    
    MTLRegion region = MTLRegionMake3D(0, 0, 0, -1, 1, -1);
    for (int i = 0; i < _transTensorShape.row; i++) {
        //  Data of each channel multipy its respect weight.
        [_multiply setSecondaryOffset:MPSOffsetMake(i, 0, 0)];
        [_multiply encodeToCommandBuffer:commandBuffer
                            primaryImage:_transposedTensor.content
                          secondaryImage:gradients.content
                        destinationImage:gradients0.content];
    
        //  Reduce sum the feature channels.
        [_sumRow encodeToCommandBuffer:commandBuffer
                           sourceImage:gradients0.content
                      destinationImage:gradients1.content];
    
        region.origin.y = i;
        [_neuronScale setClipRect:region];
        [_neuronScale encodeToCommandBuffer:commandBuffer
                                sourceImage:gradients1.content
                           destinationImage:transposedResult.content];
    }
    [gradients unlock];
    [gradients1 unlock];
    [gradients0 unlock];
    [_transposedTensor unlock];
    _transposedTensor = nil;
    
    //  Now we transpose the gradients back to the input shape.
    //  CHW -> HWC
    MetalMatrix matrix = [[MTTensorCache sharedCache] fetchMatrixWithRows:1
                                                              columns:Product(inputShape)
                                                           dataFormat:MPSDataTypeFloat16
                                                        commandBuffer:commandBuffer];
    [_imageToMatrix1 encodeToCommandBuffer:commandBuffer
                               sourceImage:transposedResult.content
                         destinationMatrix:matrix.content];
    
    MetalTensor result = [[MTTensorCache sharedCache] fetchTensorWithShape:inputShape dataType:_dataType commandBuffer:commandBuffer];
    [_matrixToImage1 encodeToCommandBuffer:commandBuffer
                              sourceMatrix:matrix.content
                          destinationImage:result.content];
    [matrix unlock];
    [transposedResult unlock];
    
    [self removeGradient];
    [self removeCachedImages];
    
    if (self.stopGradient) {
        [self.blit encodeToCommandBuffer:commandBuffer
                             sourceImage:result.content
                        destinationImage:self.savedGradients.content];
        [result unlock];
    }
    else {
        [backwardTarget setGradient:result forwardTarget:self];
        [result unlock];
        [backwardTarget gradientReadyOnCommandBuffer:commandBuffer forwardTarget:self];
    }
}
*/

@end
