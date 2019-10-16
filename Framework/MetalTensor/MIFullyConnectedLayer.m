//
//  MIFullyConnectedLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIFullyConnectedLayer.h"
#import "MITemporaryImageCache.h"
#import "MIDataSource.h"

@interface MIFullyConnectedLayer() {
    MPSCNNFullyConnected *_fullyConnected;
}

@end

@implementation MIFullyConnectedLayer

- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
    NSParameterAssert(_kernel.column == _inputShapes[0].column &&
                      _kernel.row == _inputShapes[0].row &&
                      _kernel.depth == _inputShapes[0].depth);
    
    _outputShape.column = 1;
    _outputShape.row = 1;
    _outputShape.depth = _kernel.filters;
    
    if (_dataSource) {
        _fullyConnected = [[MPSCNNFullyConnected alloc] initWithDevice:_device weights:_dataSource];
    }
}

- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    NSAssert(_dataSource != nil, @"The weights has not been set.");

    DB_TRACE(-_verbose+2, "\n%s encoding...", self.labelUTF8);
    
    _outputTempImage = [[MITemporaryImageCache sharedCache] fetchTemporaryImageWithShape:&_outputShape commandBuffer:commandBuffer];
    [_outputTempImage newTemporaryImageForCommandBuffer:commandBuffer];
    [_fullyConnected encodeToCommandBuffer:commandBuffer
                              sourceImage:_inputs[@(0)].image
                         destinationImage:_outputTempImage.image];
    
    [self removeCachedImages];
    
    [self notifyTargetsAboutNewTempImage:commandBuffer];
}

#pragma mark - Management of the weights

- (BOOL)didLoadWeights {
    return self.dataSource != nil;
}

- (void)loadWeights {
    NSAssert(_dataSource, @"The weights data source object is not initialized yet.");
    [self.dataSource load];
}

- (void)loadWeights:(NSString *)weights range:(NSRange *)range {
    self.dataSource = MakeDataSource2(weights, &_kernel, &_neuron, NO, &range[0]);;
    [self.dataSource load];
}

- (void)setDataSource:(MICNNKernelDataSource *)dataSource {
    
    NSParameterAssert(dataSource);
    
    _dataSource = dataSource;
    
    if (_device) {
        _fullyConnected = [[MPSCNNFullyConnected alloc] initWithDevice:_device weights:_dataSource];
    }
    DB_TRACE(-_verbose+1, "\n%s data source --> %s", self.labelUTF8, [dataSource description].UTF8String);
}

@end
