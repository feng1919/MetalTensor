//
//  MIMSELayer.m
//  MetalTensor
//
//  Created by Feng Stone on 2020/1/1.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import "MIMSELayer.h"
#import "MITemporaryImageCache.h"

@implementation MIMSELayer
{
@private
    DataShape _outputArithmetic;
    DataShape _outputReduceRowMean;
    DataShape _outputReduceColumnMean;
    
    MPSCNNSubtract *_subtract;
    MPSCNNMultiply *_multiply;
    MPSNNReduceRowMean *_reduceRowMean;
    MPSNNReduceColumnMean *_reduceColumnMean;
    MPSNNReduceFeatureChannelsMean *_reduceDepthMean;
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    
    NSAssert(_numOfInputs == 2, @"Invalid number of inputs, it must be two inputs.");
    NSAssert(DataShapesTheSame(&_inputShapes[0], &_inputShapes[1]), @"The two input tensors must have same shape.");
    
    _outputShape = DataShapeMake(1, 1, 1); // Output one scalar.
    _outputArithmetic = _inputShapes[0];
    _outputReduceRowMean = DataShapeMake(1, _inputShapes[0].column, _inputShapes[0].depth);
    _outputReduceColumnMean = DataShapeMake(1, 1, _inputShapes[0].depth);
    
    _subtract = [[MPSCNNSubtract alloc] initWithDevice:device];
    _multiply = [[MPSCNNMultiply alloc] initWithDevice:device];
    _reduceRowMean = [[MPSNNReduceRowMean alloc] initWithDevice:device];
    _reduceColumnMean = [[MPSNNReduceColumnMean alloc] initWithDevice:device];
    _reduceDepthMean = [[MPSNNReduceFeatureChannelsMean alloc] initWithDevice:device];
}

- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);
    
    MITemporaryImage *subtractImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputArithmetic commandBuffer:commandBuffer];
    [subtractImage newTemporaryImageForCommandBuffer:commandBuffer];
    MITemporaryImage *multiplyImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputArithmetic commandBuffer:commandBuffer];
    [multiplyImage newTemporaryImageForCommandBuffer:commandBuffer];
    MITemporaryImage *reduceRowImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputReduceRowMean commandBuffer:commandBuffer];
    [reduceRowImage newTemporaryImageForCommandBuffer:commandBuffer];
    MITemporaryImage *reduceColumnImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputReduceColumnMean commandBuffer:commandBuffer];
    [reduceColumnImage newTemporaryImageForCommandBuffer:commandBuffer];
    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:commandBuffer];
    [_outputTempImage newTemporaryImageForCommandBuffer:commandBuffer];
    
    [_subtract encodeToCommandBuffer:commandBuffer
                        primaryImage:_inputs[@(0)].image
                      secondaryImage:_inputs[@(1)].image
                    destinationImage:subtractImage.image];
    [_multiply encodeToCommandBuffer:commandBuffer
                        primaryImage:subtractImage.image
                      secondaryImage:subtractImage.image
                    destinationImage:multiplyImage.image];
    [_reduceRowMean encodeToCommandBuffer:commandBuffer
                              sourceImage:multiplyImage.image
                         destinationImage:reduceRowImage.image];
    [_reduceColumnMean encodeToCommandBuffer:commandBuffer
                                 sourceImage:reduceRowImage.image
                            destinationImage:reduceColumnImage.image];
    [_reduceDepthMean encodeToCommandBuffer:commandBuffer
                                sourceImage:reduceColumnImage.image
                           destinationImage:_outputTempImage.image];
    [subtractImage unlock];
    [multiplyImage unlock];
    [reduceRowImage unlock];
    [reduceColumnImage unlock];
    
    [self removeCachedImages];
    [self notifyTargetsAboutNewTempImage:commandBuffer];
}

@end
