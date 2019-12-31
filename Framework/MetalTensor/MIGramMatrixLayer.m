//
//  MIGramMatrixLayer.m
//  MetalTensor
//
//  Created by Feng Stone on 2019/12/31.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIGramMatrixLayer.h"
#import "MITemporaryImageCache.h"

@implementation MIGramMatrixLayer {
@private
    MPSCNNMultiply *_arithmetic;
    MPSNNReduceRowMean *_reduceRow;
    MPSNNReduceColumnMean *_reduceColumn;
    
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
    
    _arithmetic = [[MPSCNNMultiply alloc] initWithDevice:device];
    _arithmetic.primaryScale = 1.0f;
    _arithmetic.bias = 0.0f;
    _arithmetic.primaryStrideInPixelsX = 1;
    _arithmetic.primaryStrideInPixelsY = 1;
    _arithmetic.primaryStrideInFeatureChannels = 1;
    _arithmetic.secondaryScale = 1.0f;
    _arithmetic.secondaryStrideInPixelsX = 1;
    _arithmetic.secondaryStrideInPixelsY = 1;
    _arithmetic.secondaryStrideInFeatureChannels = 0;
    _arithmetic.destinationFeatureChannelOffset = 0;
    
    _reduceRow = [[MPSNNReduceRowMean alloc] initWithDevice:device];
    _reduceColumn = [[MPSNNReduceColumnMean alloc] initWithDevice:device];
}

- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);
    
    int numOfChannels = _inputShapes[0].depth;
    MITemporaryImage *arithmeticImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputArithmetic commandBuffer:commandBuffer];
    [arithmeticImage newTemporaryImageForCommandBuffer:commandBuffer];
    MITemporaryImage *reduceRowImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputReduceRow commandBuffer:commandBuffer];
    [reduceRowImage newTemporaryImageForCommandBuffer:commandBuffer];
    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:commandBuffer];
    [_outputTempImage newTemporaryImageForCommandBuffer:commandBuffer];
    
    MPSOffset secondaryOffSet = {0, 0, 0};
    MTLRegion clipRect;
    clipRect.origin = MTLOriginMake(0, 0, 0);
    clipRect.size = MTLSizeMake(_inputShapes[0].column, _inputShapes[0].row, _inputShapes[0].depth);
    for (int i = 0; i < numOfChannels; i++) {
        secondaryOffSet.z = i;
        [_arithmetic setSecondaryOffset:secondaryOffSet];
        [_arithmetic encodeToCommandBuffer:commandBuffer
                              primaryImage:_inputs[@(0)].image
                            secondaryImage:_inputs[@(0)].image
                          destinationImage:arithmeticImage.image];
        [_reduceRow encodeToCommandBuffer:commandBuffer
                              sourceImage:arithmeticImage.image
                         destinationImage:reduceRowImage.image];
        clipRect.origin.y = i;
        [_reduceColumn setClipRect:clipRect];
        [_reduceColumn encodeToCommandBuffer:commandBuffer
                                 sourceImage:reduceRowImage.image
                            destinationImage:_outputTempImage.image];
        
    }
    [arithmeticImage unlock];
    [reduceRowImage unlock];
    [self removeCachedImages];
    [self notifyTargetsAboutNewTempImage:commandBuffer];
}


@end
