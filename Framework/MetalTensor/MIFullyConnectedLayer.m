//
//  MIFullyConnectedLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIFullyConnectedLayer.h"
#import <MetalImage/MetalDevice.h>
#import "MITemporaryImageCache.h"
#import "MIDataSource.h"

@interface MIFullyConnectedLayer() {
    MPSCNNFullyConnected *_fullyConnected;
}

@end

@implementation MIFullyConnectedLayer

- (instancetype)initWithInputShape:(DataShape *)inputShape
                       outputShape:(DataShape *)outputShape
                  kernelDataSource:(id<MPSCNNConvolutionDataSource>)dataSource {
    if (self = [super initWithInputShape:inputShape outputShape:outputShape]) {
        _dataSource = dataSource;
    }
    return self;
}

- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    NSAssert(_dataSource != nil, @"The weights has not been set.");
    if (_fullyConnected == nil) {
        _fullyConnected = [[MPSCNNFullyConnected alloc] initWithDevice:[MetalDevice sharedMTLDevice]
                                                              weights:_dataSource];
    }
    
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

- (void)loadWeights:(NSString *)weights range:(NSRange *)range
        kernelShape:(KernelShape *)k neuronType:(NeuronType *)n depthWise:(BOOL)depthWise {
    self.dataSource = MakeDataSource2(weights, k, n, depthWise, &range[0]);;
    [self.dataSource load];
}

- (void)setDataSource:(id<MPSCNNConvolutionDataSource>)dataSource {
    _dataSource = dataSource;
    DB_TRACE(-_verbose+1, "\n%s data source --> %s", self.labelUTF8, [dataSource description].UTF8String);
}

@end
