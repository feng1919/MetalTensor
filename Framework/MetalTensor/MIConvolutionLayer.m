//
//  MIConvolutionLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIConvolutionLayer.h"
#import "MTTensorCache.h"
#import "MIDataSource.h"

@interface MIConvolutionLayer() {
    
    MPSCNNConvolution *_convolution;
    
    MPSCNNNeuron *_neuronOp;
    MPSCNNNeuronGradient *_neuronGradient;
    MetalTensor _convTensor;
    
    MICNNKernelDataSource *_backwardDataSource;
    MPSCNNConvolutionTranspose *_convTranspose;
}

@property (nonatomic, weak) BackwardTarget backwardTarget;

@end

@implementation MIConvolutionLayer

#pragma mark - override

- (void)initialize {
    _edgeMode = MPSImageEdgeModeZero;
    _depthWise = NO;
    _neuron.neuron = MPSCNNNeuronTypeNone;
    _neuron.a = 0.0f;
    _neuron.b = 0.0f;
    _padding = MTPaddingMode_tfsame;
    _offset.x = 0;
    _offset.y = 0;
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    [self updateComputing];
}

- (void)updateOutputShape {
    
    if (_device) {
        _outputShape.column = conv_output_length(_inputShapes[0].column, _kernel.column, _kernel.stride, _padding);
        _outputShape.row = conv_output_length(_inputShapes[0].row, _kernel.row, _kernel.stride, _padding);
        _outputShape.depth = _depthWise?_inputShapes[0].depth:_kernel.filters;
    }
}

#pragma mark - public

- (void)setEdgeMode:(MPSImageEdgeMode)edgeMode {
    _edgeMode = edgeMode;
    [_convolution setEdgeMode:_edgeMode];
}

- (void)setOffset:(MTLInt2)offset {
    _offset = offset;

    MPSOffset mpsOffset;
    mpsOffset.x = _offset.x;
    mpsOffset.y = _offset.y;
    mpsOffset.z = 0;
    [_convolution setOffset:mpsOffset];
}

- (void)setDataSource:(MICNNKernelDataSource *)dataSource {
    
    NSParameterAssert(dataSource);
    
    _dataSource = dataSource;
    
    DB_TRACE(-_verbose+1, "\n%s data source --> %s", self.labelUTF8, [dataSource description].UTF8String);
    
    [self updateComputing];
}

- (void)updateComputing {
    
    if (_device && _dataSource) {
        
        if (_needBackward) {
            
            /*
             *  There is a bug of MPSCNNConvolutionGradient kernel, it does not handle activation,
             *  so we depend on ourselves.
             *
             */
            
            _dataSource.neuron = NeuronTypeMake(MPSCNNNeuronTypeNone, 0.0f, 0.0f);
            
            _convolution = [[MPSCNNConvolution alloc] initWithDevice:_device weights:_dataSource];
            [_convolution setEdgeMode:_edgeMode];
            [self setOffset:_offset];
            
            _backwardDataSource = [_dataSource copy];
            _backwardDataSource.neuron = NeuronTypeMake(MPSCNNNeuronTypeNone, 0.0f, 0.0f);
            _backwardDataSource.rotateSpatial180 = YES;
            _backwardDataSource.transposeIO = YES;
            _backwardDataSource.removeBias = YES;
            
            _convTranspose = [[MPSCNNConvolutionTranspose alloc] initWithDevice:_device weights:_backwardDataSource];
            _convTranspose.kernelOffsetX = trans_conv_offset(_kernel.column, _kernel.stride, _padding);
            _convTranspose.kernelOffsetY = trans_conv_offset(_kernel.row, _kernel.stride, _padding);
            _convTranspose.edgeMode = _edgeMode;
            
            MPSNNNeuronDescriptor *neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:_neuron.neuron
                                                                                                 a:_neuron.a
                                                                                                 b:_neuron.b
                                                                                                 c:_neuron.c];
            _neuronOp = [[MPSCNNNeuron alloc] initWithDevice:_device neuronDescriptor:neuronDesc];
            _neuronGradient = [[MPSCNNNeuronGradient alloc] initWithDevice:_device neuronDescriptor:neuronDesc];
        }
        else {

            _convolution = [[MPSCNNConvolution alloc] initWithDevice:_device weights:_dataSource];
            [_convolution setEdgeMode:_edgeMode];
            [self setOffset:_offset];
        }
    }
}

- (void)processImagesOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+3, "\n%s encoding...", self.labelUTF8);

    NSAssert(_inputImages.count > 0, @"There is no input image received.");

    if (_needBackward) {
        
        MetalTensor sourceTensor = _inputImages[@(0)];
        self.backwardTarget = sourceTensor.source;
        
        _convTensor = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape commandBuffer:commandBuffer];
        _image = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape commandBuffer:commandBuffer];
        _image.source = self;
        
        [_convolution encodeToCommandBuffer:commandBuffer
                              sourceImage:sourceTensor.content
                         destinationImage:_convTensor.content];
        
        _state = [_neuronOp temporaryResultStateForCommandBuffer:commandBuffer
                                                     sourceImage:_convTensor.content
                                                    sourceStates:nil
                                                destinationImage:_image.content];
        [_neuronOp encodeToCommandBuffer:commandBuffer
                             sourceImage:_convTensor.content
                        destinationState:_state
                        destinationImage:_image.content];
    }
    else {
        _image = [[MTTensorCache sharedCache] fetchTensorWithShape:&_outputShape commandBuffer:commandBuffer];
        _image.source = self;

        MetalTensor sourceTensor = _inputImages[@(0)];

        [_convolution encodeToCommandBuffer:commandBuffer
                              sourceImage:sourceTensor.content
                         destinationImage:_image.content];

    }
    
    [self removeCachedImages];
}

- (void)processGradientsOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {

    BackwardTarget backwardTarget = self.backwardTarget;
    NSAssert(backwardTarget, @"Invalid backward connection...");

    MetalTensor activatedTensor = [[MTTensorCache sharedCache] fetchTensorWithShape:_gradient.shape
                                                                      commandBuffer:commandBuffer];
    [_neuronGradient encodeToCommandBuffer:commandBuffer
                            sourceGradient:_gradient.content
                               sourceImage:_convTensor.content
                             gradientState:_state
                       destinationGradient:activatedTensor.content];
    [_convTensor unlock];

    MetalTensor destinationGradient = [[MTTensorCache sharedCache] fetchTensorWithShape:&_inputShapes[0]
                                                                          commandBuffer:commandBuffer];

    [_convTranspose encodeToCommandBuffer:commandBuffer
                              sourceImage:activatedTensor.content
                         destinationImage:destinationGradient.content];
    
    [activatedTensor unlock];
    
    [self removeState];
    [self removeCachedImages];
    [self removeGradient];
    
    if (self.stopGradient) {
        [self.blit encodeToCommandBuffer:commandBuffer
                             sourceImage:destinationGradient.content
                        destinationImage:self.savedGradients.content];
        [destinationGradient unlock];
    }
    else {
        [backwardTarget setGradient:destinationGradient forwardTarget:self];
        [destinationGradient unlock];
        [backwardTarget gradientReadyOnCommandBuffer:commandBuffer forwardTarget:self];
    }
}

#pragma mark - Management of the weights

- (BOOL)didLoadWeights {
    return self.dataSource != nil;
}

- (void)loadWeights {
    NSAssert(_dataSource, @"The weights data source object is not initialized yet.");
    [self.dataSource load];
}

- (void)loadWeights:(NSData *)weights {
    self.dataSource = [[MICNNKernelDataSource alloc] initWithData:weights
                                                           kernel:&_kernel
                                                           neuron:&_neuron
                                                        depthWise:_depthWise];
    [self.dataSource load];
}

- (void)loadWeights:(NSString *)weights range:(NSRange *)range {
    self.dataSource = MakeDataSource2(weights, &_kernel, &_neuron, _depthWise, &range[0]);;
    [self.dataSource load];
}

MIConvolutionLayer *MakeConvolutionLayer(NSString *module_name,
                                         KernelShape *k,
                                         NeuronType *n,
                                         MTPaddingMode padding,
                                         DataShape *input)
{
    MICNNKernelDataSource *data_cnn = MakeDataSource(module_name, k, n);
    MIConvolutionLayer *module = [[MIConvolutionLayer alloc] initWithInputShape:input];
    module.kernel = *k;
    module.neuron = *n;
    module.padding = padding;
    module.dataSource = data_cnn;
    module.label = module_name;
    return module;
}

@end
