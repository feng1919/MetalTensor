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
    MPSCNNNeuron *_alphaNeuron;
    MPSCNNPoolingAverage *_pooling;
    MTChannelReduce *_channelReduceMean;
}

- (instancetype)initWithInputShape:(DataShape *)dataShape {
    DataShape *inputShapes[2] = {dataShape, dataShape};
    if (self = [super initWithInputShapes1:inputShapes size:2]) {
        
    }
    return self;
}

#pragma mark - override

- (void)initialize {
    _alpha = 1.0f;
}

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
    _pooling.offset = MPSOffsetMake(inputShape->column>>1, inputShape->row>>1, 0);
    
    _channelReduceMean = [[MTChannelReduce alloc] initWithReduceType:ReduceTypeMean numberOfChannels:inputShape->depth];
    _channelReduceMean.alpha = _alpha;
    [_channelReduceMean compile:device];
    
    if (_needBackward) {
        MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeLinear
                                                                                             a:-1.0f
                                                                                             b:0.0f
                                                                                             c:0.0f];
        _negative = [[MPSCNNNeuron alloc] initWithDevice:device neuronDescriptor:neuronDesc];
        
        neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeLinear
                                                                      a:_alpha/(float)Product(inputShape)
                                                                      b:0.0f
                                                                      c:0.0f];
        _alphaNeuron = [[MPSCNNNeuron alloc] initWithDevice:device neuronDescriptor:neuronDesc];
    }
}

- (void)updateOutputShape {
    if (_device) {

        _outputShape = DataShapeMake(1, 1, 1); // Output one scalar.
        _outputArithmetic = _inputShapes[0];
        _outputPooling = DataShapeMake(1, 1, _inputShapes[0].depth);
    }
}

- (void)setSecondaryImage:(MTImageTensor *)secondaryImage {
    _secondaryImage = secondaryImage;
    if (secondaryImage) {
        [self reserveImageIndex:1];
    }
    else {
        [self releaseImageIndex:1];
    }
}

#pragma mark - MTTensorForward delegate

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

- (void)setImage:(MetalTensor)newImage atIndex:(NSInteger)imageIndex {
    NSAssert(DataShapesTheSame(newImage.shape, &_inputShapes[0]), @"Invalid input tensor shape.");
    [super setImage:newImage atIndex:imageIndex];
    if (_secondaryImage) {
        [super setImage:_secondaryImage atIndex:1];
    }
}

- (void)imageReadyOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer atIndex:(NSInteger)imageIndex {
    [super imageReadyOnCommandBuffer:commandBuffer atIndex:imageIndex];
    if (_secondaryImage) {
        [super imageReadyOnCommandBuffer:commandBuffer atIndex:1];
    }
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    /*
     *  MSE f = 0.5*mean((v0-v1)^2).
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

#if DEBUG
    if (self.dumpResult) {
        [self saveTensor:_image onCommandBuffer:commandBuffer];
    }
#endif
}

#pragma mark - MTTensorBackward Delegate
- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    /*
     *  MSE f = 0.5*mean((v0-v1)^2) = 0.5*mean(v0^2-2*v0*v1+v1^2).
     *  The derivative of v0: df/dv0 = (v0-v1)/volume.
     *  The derivative of v1: df/dv1 = (v1-v0)/volume = -(df/dv0).
     */
    
    MetalTensor t0 = _inputImages[@(0)];
    MetalTensor t1 = _inputImages[@(1)];
    BackwardTarget back0 = t0.source;
    NSAssert(back0, @"Invalid primary backward target...");
    BackwardTarget back1 = t1.source;
//    NSAssert(back1, @"Invalid secondary backward target...");
    
    MetalTensor dv0 = [[MTTensorCache sharedCache] fetchTensorWithShape:t0.shape commandBuffer:commandBuffer];
    MetalTensor dv1 = [[MTTensorCache sharedCache] fetchTensorWithShape:t1.shape commandBuffer:commandBuffer];
    
    DB_TRACE(-_verbose+2, "\n%s backward encoding...", self.labelUTF8);
    
    //  The derivative of v0.
    [_subtract encodeToCommandBuffer:commandBuffer
                        primaryImage:t0.content
                      secondaryImage:t1.content
                    destinationImage:dv1.content];
    [_alphaNeuron encodeToCommandBuffer:commandBuffer
                            sourceImage:dv1.content
                       destinationImage:dv0.content];
    
    [self removeGradient];
    [back0 setGradient:dv0 forwardTarget:self];
    
    if (back1) {
        [_negative encodeToCommandBuffer:commandBuffer
                             sourceImage:dv0.content
                        destinationImage:dv1.content];
        [back1 setGradient:dv1 forwardTarget:self];
    }
    
    [self removeCachedImages];
    
    [dv0 unlock];
    [dv1 unlock];
    
    [back0 gradientReadyOnCommandBuffer:commandBuffer forwardTarget:self];
    [back1 gradientReadyOnCommandBuffer:commandBuffer forwardTarget:self];
    
}

@end
