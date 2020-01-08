//
//  MIFullyConnectedLayer.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIFullyConnectedLayer.h"
#import "MTTensorCache.h"
#import "MIDataSource.h"

@interface MIFullyConnectedLayer() {
    MPSCNNFullyConnected *_fullyConnected;
    MPSCNNFullyConnectedGradient *_fcGradientOp;
}

@end

@implementation MIFullyConnectedLayer

- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
    NSParameterAssert(_kernel.column == _inputShapes[0].column &&
                      _kernel.row == _inputShapes[0].row &&
                      _kernel.depth == _inputShapes[0].depth);
    
    _dataShape.column = 1;
    _dataShape.row = 1;
    _dataShape.depth = _kernel.filters;
    
    [self updateComputing];
}

- (void)updateComputing {
    
    if (_dataSource && _device) {
        _fullyConnected = [[MPSCNNFullyConnected alloc] initWithDevice:_device weights:_dataSource];
        if (_needBackward) {
            _fcGradientOp = [[MPSCNNFullyConnectedGradient alloc] initWithDevice:_device weights:_dataSource];
            _fcGradientOp.gradientOption = MPSCNNConvolutionGradientOptionGradientWithData;
        }
        _operation = _fullyConnected;
        _gradientOp = _fcGradientOp;
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
