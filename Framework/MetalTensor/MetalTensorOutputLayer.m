//
//  MetalTensorOutputLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/5.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorOutputLayer.h"

@interface MetalTensorOutputLayer() {
    MPSCNNNeuron *_neuron;
}

@end

@implementation MetalTensorOutputLayer

#pragma mark - override
- (void)initialize {
    _dataFormat = TensorDataFormatFloat16;
    _neuronType.neuron = MPSCNNNeuronTypeNone;
    _neuronType.a = 0.0f;
    _neuronType.b = 0.0f;
    _neuronType.c = 0.0f;
}

- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
    /*
     *  The neuron operation just does nothing but copying tensor
     *  from GPU space to CPU space, so we may access it.
     *  TODO: DMA optimization
     */
    
    MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:_neuronType.neuron
                                                                                         a:_neuronType.a
                                                                                         b:_neuronType.b
                                                                                         c:_neuronType.c];
    _neuron = [[MPSCNNNeuron alloc] initWithDevice:device neuronDescriptor:neuronDesc];
    
    MPSImageDescriptor *desc = ImageDescriptor(&_outputShape, _dataFormat);
    desc.storageMode = MTLStorageModeShared;
    _outputImage = [[MPSImage alloc] initWithDevice:_device imageDescriptor:desc];
    
}

- (void)notifyTargetsAboutNewImageOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {

}

#pragma mark - MTTensorForward Delegate
- (void)setInputShape:(DataShape *)dataShape atIndex:(NSInteger)imageIndex {
    [super setInputShape:dataShape atIndex:imageIndex];
    
    if (_device) {
        MPSImageDescriptor *desc = ImageDescriptor(&_outputShape, _dataFormat);
        desc.storageMode = MTLStorageModeShared;
        _outputImage = [[MPSImage alloc] initWithDevice:_device imageDescriptor:desc];
    }
}

- (void)setImage:(MetalTensor)newImage atIndex:(NSInteger)imageIndex {
    NSAssert(DataShapesTheSame(newImage.shape, &_inputShapes[0]), @"Invalid input tensor shape.");
    [super setImage:newImage atIndex:imageIndex];
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    NSAssert(_neuron, @"The layer %@ is not compiled.", self.label);
    DB_TRACE(-_verbose+2, "\n%s forward encoding...", self.labelUTF8);
    [_neuron encodeToCommandBuffer:commandBuffer
                       sourceImage:_inputImages[@(0)].content
                  destinationImage:_outputImage];
    
    if (!_needBackward) {
        [self removeCachedImages];
    }
}

#pragma mark - MTTensorBackward Delegate

- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+2, "\n%s backward encoding...", self.labelUTF8);
    
    MetalTensor sourceTensor = _inputImages[@(0)];
    [sourceTensor.source setGradient:sourceTensor forwardTarget:self];
    [self removeCachedImages];
    [self removeGradient];
    
    [sourceTensor.source gradientReadyOnCommandBuffer:commandBuffer forwardTarget:self];
}

#pragma mark - DEBUG
#ifdef DEBUG
- (void)debugOutputTensorWithCommandBuffer:(id<MTLCommandBuffer>)command_buffer {
    
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    
    float16_t *result = malloc(ProductOfDataShape(&_outputShape)*sizeof(float16_t));
    [_outputImage toFloat16Array:result];
    // Once we obtain the tensor buffer, we may do something blah blah blah...
    free(result);
}
#endif

@end
