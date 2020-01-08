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

- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
    /*
     *  The neuron operation just does nothing but copying tensor
     *  from GPU space to CPU space, so we may access it.
     *  TODO: DMA optimization
     */
    
    MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeNone];
    _neuron = [[MPSCNNNeuron alloc] initWithDevice:device neuronDescriptor:neuronDesc];
    
    MPSImageDescriptor *desc = ImageDescriptor(&_outputShape);
    desc.storageMode = MTLStorageModeShared;
    _outputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];
    
}

- (DataShape *)dataShapeRef {
    return &_outputShape;
}

- (void)setImage:(MetalTensor)newImage atIndex:(NSInteger)imageIndex {
    NSAssert(DataShapesTheSame(newImage.shape, &_inputShapes[0]), @"Invalid input tensor shape.");
    [super setImage:newImage atIndex:imageIndex];
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+2, "\n%s forward encoding...", self.labelUTF8);
    [_neuron encodeToCommandBuffer:commandBuffer
                       sourceImage:_inputImages[@(0)].content
                  destinationImage:_outputImage];
    
    if (!_needBackward) {
        [self removeCachedImages];
    }
}

- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+2, "\n%s backward encoding...", self.labelUTF8);
    
    MetalTensor sourceTensor = _inputImages[@(0)];
    [sourceTensor.source setGradient:sourceTensor forwardTarget:self];
    [self removeCachedImages];
    [self removeGradient];
    [sourceTensor.source gradientReadyFromForwardTarget:self onCommandBuffer:commandBuffer];
    
}

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
