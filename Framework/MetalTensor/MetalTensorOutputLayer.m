//
//  MetalTensorOutputLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/5.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorOutputLayer.h"
#import <MetalImage/MetalDevice.h>
#import "MITemporaryImageCache.h"

@interface MetalTensorOutputLayer() {
    MPSCNNNeuron *_neuron;
}

@end

@implementation MetalTensorOutputLayer

- (instancetype)initWithOutputShape:(DataShape *)dataShape1 {
    if (self = [super initWithInputShape:dataShape1 outputShape:dataShape1]) {
        [self initializeWithDataShape:dataShape1];
        DB_TRACE(-_verbose+2, "\n%s init --> %s", self.labelUTF8, NSStringFromDataShape(dataShape1));
    }
    return self;
}

- (void)initializeWithDataShape:(DataShape *)dataShape {
    
    /*
     *  It just does nothing.
     *  Copying tensor from GPU space to CPU space, so we may access it.
     *  TODO: DMA optimization
     */
    
    MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeNone];
    _neuron = [[MPSCNNNeuron alloc] initWithDevice:[MetalDevice sharedMTLDevice] neuronDescriptor:neuronDesc];
    
    MPSImageDescriptor *desc = ImageDescriptor(dataShape);
    desc.storageMode = MTLStorageModeShared;
    _outputImage = [[MPSImage alloc] initWithDevice:[MetalDevice sharedMTLDevice]
                                    imageDescriptor:desc];
    
}

- (void)setInputImage:(MITemporaryImage *)newInputImage atIndex:(NSInteger)imageIndex {
    NSAssert(DataShapesTheSame(newInputImage.shape, &_inputShapes[0]), @"Invalid input tensor shape.");
    [super setInputImage:newInputImage atIndex:imageIndex];
}

- (void)processTensorWithCommandBuffer:(id<MTLCommandBuffer>)cmdBuf {
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);
    [_neuron encodeToCommandBuffer:cmdBuf
                      sourceImage:_inputs[@(0)].image
                 destinationImage:_outputImage];
    [self removeCachedImages];
}

#ifdef DEBUG
- (void)debugOutputTensorWithCommandBuffer:(id<MTLCommandBuffer>)command_buffer {
    
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    
    float16_t *result = malloc(_outputShape.row*_outputShape.column*_outputShape.depth*sizeof(float16_t));
    [_outputImage toFloat16Array:result];
    // Once we obtain the tensor buffer, we may do something blah blah blah...
    free(result);
}
#endif

@end
