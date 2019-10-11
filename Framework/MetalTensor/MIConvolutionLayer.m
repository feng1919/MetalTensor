//
//  MIConvolutionLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIConvolutionLayer.h"
#import <MetalImage/MetalDevice.h>
#import "MITemporaryImageCache.h"
#import "MIDataSource.h"

@interface MIConvolutionLayer() {
    
    MPSCNNConvolution *_convolution;
}

@end

@implementation MIConvolutionLayer

- (instancetype)initWithInputShape:(DataShape *)inputShape outputShape:(DataShape *)outputShape {
    if (self = [super initWithInputShape:inputShape outputShape:outputShape]) {
        NSAssert(DataShapeValid(inputShape), @"Invalid input shape...");
        NSAssert(DataShapeValid(outputShape), @"Invalid output shape...");
        _edgeMode = MPSImageEdgeModeZero;
        _offset.x = 0;
        _offset.y = 0;
        _offset.z = 0;
        
        DB_TRACE(-_verbose+2, "\n%s init %s --> %s", self.labelUTF8, NSStringFromDataShape(inputShape).UTF8String,
                 NSStringFromDataShape(outputShape).UTF8String);
    }
    return self;
}

- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape
                  kernelDataSource:(id<MPSCNNConvolutionDataSource>)dataSource {
    if (self = [super initWithInputShape:inputShape outputShape:outputShape]) {
        NSAssert(DataShapeValid(inputShape), @"Invalid input shape...");
        NSAssert(DataShapeValid(outputShape), @"Invalid output shape...");
        _edgeMode = MPSImageEdgeModeZero;
        _offset.x = 0;
        _offset.y = 0;
        _offset.z = 0;
        
        DB_TRACE(-_verbose+2, "\n%s init %s --> %s", self.labelUTF8, NSStringFromDataShape(inputShape).UTF8String,
                 NSStringFromDataShape(outputShape).UTF8String);
        
        self.dataSource = dataSource;
    }
    return self;
}

- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    NSAssert(_dataSource != nil, @"The weights has not been set.");
    if (_convolution == nil) {
        _convolution = [[MPSCNNConvolution alloc] initWithDevice:[MetalDevice sharedMTLDevice]
                                                        weights:_dataSource];
        _convolution.edgeMode = _edgeMode;
        [_convolution setOffset:_offset];
    }
    
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);
    
    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:commandBuffer];
    [_outputTempImage newTemporaryImageForCommandBuffer:commandBuffer];
    [_convolution encodeToCommandBuffer:commandBuffer
                           sourceImage:_inputs[@(0)].image
                      destinationImage:_outputTempImage.image];
    
    [self removeCachedImages];
    
    [self notifyTargetsAboutNewTempImage:commandBuffer];
}

- (void)setOffsetWithX:(NSUInteger)x Y:(NSUInteger)y Z:(NSUInteger)z {
    _offset.x = x;
    _offset.y = y;
    _offset.z = z;
    [_convolution setOffset:_offset];
}

- (void)setOffset:(MPSOffset)offset {
    _offset = offset;
    [_convolution setOffset:_offset];
}

- (void)setEdgeMode:(MPSImageEdgeMode)edgeMode {
    _edgeMode = edgeMode;
    [_convolution setEdgeMode:_edgeMode];
}

- (void)setDataSource:(id<MPSCNNConvolutionDataSource>)dataSource {
    _dataSource = dataSource;
    
    DB_TRACE(-_verbose+1, "\n%s data source --> %s", self.labelUTF8, [dataSource description].UTF8String);
}

#pragma mark - Management of the weights

- (BOOL)didLoadWeights {
    return self.dataSource != nil;
}

- (void)loadWeights {
    NSAssert(_dataSource, @"The weights data source object is not initialized yet.");
    [self.dataSource load];
}

- (void)loadWeights:(NSData *)weights
        kernelShape:(KernelShape *)kernelShape
         neuronType:(NeuronType *)neuronType
          depthWise:(BOOL)depthWise {
    self.dataSource = [[MICNNKernelDataSource alloc] initWithData:weights kernel:kernelShape neuron:neuronType depthWise:depthWise];
    [self.dataSource load];
}

- (void)loadWeights:(NSString *)weights range:(NSRange *)range
        kernelShape:(KernelShape *)k neuronType:(NeuronType *)n depthWise:(BOOL)depthWise {
    self.dataSource = MakeDataSource2(weights, k, n, depthWise, &range[0]);;
    [self.dataSource load];
}

MIConvolutionLayer *MakeConvolutionLayer(NSString *weight,
                                         KernelShape *k,
                                         NeuronType *n,
                                         DataShape *input,
                                         DataShape *output)
{
    MICNNKernelDataSource *data_cnn = MakeDataSource(weight, k, n);
    MIConvolutionLayer *module = [[MIConvolutionLayer alloc] initWithInputShape:input
                                                                    outputShape:output
                                                               kernelDataSource:data_cnn];
    module.label = weight;
    return module;
}

@end
