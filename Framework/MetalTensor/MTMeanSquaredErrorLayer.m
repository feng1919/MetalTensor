//
//  MTMeanSquaredErrorLayer.m
//  MetalTensor
//
//  Created by Feng Stone on 2020/1/14.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import "MTMeanSquaredErrorLayer.h"
#import "MTTensorCache.h"
#import "MTChannelReduce.h"

@implementation MTMeanSquaredErrorLayer
{
@private
    DataShape _outputArithmetic;
    DataShape _outputPooling;
    
    MPSCNNSubtract *_subtract;
    MPSCNNNeuron *_power;
    MPSCNNNeuron *_negative;
    MPSCNNMultiply *_multiply;
    MPSCNNPoolingAverage *_pooling;
    MTChannelReduce *_channelReduceMean;
}

#pragma mark - override
- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    NSAssert(_numOfImages == 2, @"Invalid number of inputs, it must be two inputs.");
//    NSAssert(DataShapesTheSame(&_inputShapes[0], &_inputShapes[1]), @"The two input tensors must have same shape.");
    DataShape *inputShape = &_inputShapes[0];
    
    _subtract = [[MPSCNNSubtract alloc] initWithDevice:device];
    MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypePower
                                                                                         a:0.707107
                                                                                         b:0.0f
                                                                                         c:2.0f];
    _power = [[MPSCNNNeuron alloc] initWithDevice:device neuronDescriptor:neuronDesc];
    _pooling = [[MPSCNNPoolingAverage alloc] initWithDevice:device
                                                kernelWidth:inputShape->column
                                               kernelHeight:inputShape->row
                                            strideInPixelsX:inputShape->column
                                            strideInPixelsY:inputShape->row];
    
    _channelReduceMean = [[MTChannelReduce alloc] initWithReduceType:ReduceTypeMean numberOfChannels:inputShape->depth];
    [_channelReduceMean compile:device];
    
    if (_needBackward) {
        MPSNNNeuronDescriptor *negativeDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeLinear
                                                                                                a:-1.0f
                                                                                                b:0.0f
                                                                                                c:0.0f];
        _negative = [[MPSCNNNeuron alloc] initWithDevice:device neuronDescriptor:negativeDesc];
        
        _multiply = [[MPSCNNMultiply alloc] initWithDevice:device];
        
    }
}

- (void)updateOutputShape {
    if (_device) {

        _outputShape = DataShapeMake(1, 1, 1); // Output one scalar.
        _outputArithmetic = _inputShapes[0];
        _outputPooling = DataShapeMake(1, 1, _inputShapes[0].depth);
    }
}

#pragma mark - MTTensorForward Delegate
- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    /*
     *  MSE f = 0.5*sum((v0-v1)^2).
     */
    
    DB_TRACE(-_verbose+2, "\n%s forward encoding...", self.labelUTF8);
    
    MetalTensor subtractImage = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputArithmetic commandBuffer:commandBuffer];
    MetalTensor squaredImage = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputArithmetic commandBuffer:commandBuffer];
    MetalTensor poolingImage = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputPooling commandBuffer:commandBuffer];
    
    _image = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape commandBuffer:commandBuffer];
    _image.source = self;
    
    [_subtract encodeToCommandBuffer:commandBuffer
                        primaryImage:_inputImages[@(0)].content
                      secondaryImage:_inputImages[@(1)].content
                    destinationImage:subtractImage.content];
    [_power encodeToCommandBuffer:commandBuffer
                      sourceImage:subtractImage.content
                 destinationImage:squaredImage.content];
    [_pooling encodeToCommandBuffer:commandBuffer
                        sourceImage:squaredImage.content
                   destinationImage:poolingImage.content];
    
    [_channelReduceMean reduceOnCommandBuffer:commandBuffer
                                 sourceTensor:poolingImage
                            destinationTensor:_image];
    
    [subtractImage unlock];
    [squaredImage unlock];
    [poolingImage unlock];
    
    if (!_needBackward) {
        [self removeCachedImages];
    }
}

#pragma mark - MTTensorBackward Delegate
- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    /*
     *  MSE f = 0.5*sum((v0-v1)^2) = 0.5*sum(v0^2-2*v0*v1+v1^2).
     *  The derivative of v0: df/dv0 = v0-v1.
     *  The derivative of v1: df/dv1 = v1-v0 = -(df/dv0).
     */
    
    MetalTensor t0 = _inputImages[@(0)];
    MetalTensor t1 = _inputImages[@(1)];
    BackwardTarget back0 = t0.source;
    BackwardTarget back1 = t1.source;
    
    MetalTensor dv0 = [[MTTensorCache sharedCache] fetchTensorWithShape:t0.shape commandBuffer:commandBuffer];
    
    DB_TRACE(-_verbose+2, "\n%s backward encoding...", self.labelUTF8);
    
    //  The derivative of v0.
    [_subtract encodeToCommandBuffer:commandBuffer
                        primaryImage:t0.content
                      secondaryImage:t1.content
                    destinationImage:dv0.content];
    [_multiply encodeToCommandBuffer:commandBuffer
                        primaryImage:dv0.content
                      secondaryImage:_gradient.content
                    destinationImage:t0.content];
    [dv0 unlock];
    [self removeGradient];
    [back0 setGradient:t0 forwardTarget:self];
    
    [_negative encodeToCommandBuffer:commandBuffer
                         sourceImage:t0.content
                    destinationImage:t1.content];
    [back1 setGradient:t1 forwardTarget:self];
    [self removeCachedImages];
    
    [back0 gradientReadyOnCommandBuffer:commandBuffer forwardTarget:self];
    [back1 gradientReadyOnCommandBuffer:commandBuffer forwardTarget:self];
    
}


@end
