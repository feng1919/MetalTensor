//
//  MITransposeConvolutionLayer.m
//  MetalTensorDemo
//
//  Created by Feng Stone on 2019/9/24.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MITransposeConvolutionLayer.h"
#import <MetalImage/MetalDevice.h>
#import "MITemporaryImageCache.h"
#import "MIDataSource.h"

@interface MITransposeConvolutionLayer() {
    
    MPSCNNConvolutionTranspose *_convolution;
}

@end

@implementation MITransposeConvolutionLayer


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
        _convolution = [[MPSCNNConvolutionTranspose alloc] initWithDevice:[MetalDevice sharedMTLDevice]
                                                                  weights:_dataSource];
        _convolution.edgeMode = _edgeMode;
        [_convolution setOffset:_offset];
        [_convolution setKernelOffsetX:_kernelOffset.x];
        [_convolution setKernelOffsetY:_kernelOffset.y];
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

- (void)loadWeights:(NSString *)weights range:(NSRange *)range
        kernelShape:(KernelShape *)k neuronType:(NeuronType *)n depthWise:(BOOL)depthWise{
    self.dataSource = MakeDataSource2(weights, k, n, depthWise, &range[0]);
    [self.dataSource load];
    
    // NHWC
    // 40 * 3 * 3 * 40
    {
        MICNNKernelDataSource *dataSource = (MICNNKernelDataSource *)self.dataSource;
        NSData *data = [dataSource data];
        float *buffer = (float *)[data bytes];
//        int n = dataSource.kernel.kernel;
        int h = dataSource.kernel.row;
        int w = dataSource.kernel.column;
        int c = dataSource.kernel.depth;
        printf("\n weights: %s", self.labelUTF8);
        printf("\n c: %d", c);
        for (int i = 0; i < h; i ++) {
            printf("\n");
            for (int j = 0; j < w; j++) {
                printf("%f ", buffer[(i * w + j) * c]);
            }
        }
        printf("\n");
    }
}

@end
