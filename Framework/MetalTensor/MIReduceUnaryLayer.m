//
//  MIReduceUnaryLayer.m
//  MetalTensor
//
//  Created by Feng Stone on 2019/12/31.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIReduceUnaryLayer.h"

@implementation MIReduceUnaryLayer {
@protected
    MPSNNReduceUnary *_reduce;
}

- (void)initialize {
    _type = ReduceTypeMean;
    _axis = ReduceAxisDepth;
}

- (void)compile:(id<MTLDevice>)device {
    [super compile:device];
    
    _outputShape = _inputShapes[0];
    if (_axis == ReduceAxisDepth) {
        _outputShape.depth = 1;
    }
    else if (_axis == ReduceAxisColumn) {
        _outputShape.column = 1;
    }
    else {
        _outputShape.row = 1;
    }
    
    NSString *className = GetReduceClass(_type, _axis);
    Class c = NSClassFromString(className);
    _reduce = [[c alloc] initWithDevice:device];
    
    if (_needBackward) {
        _gradientOp = [[MPSCNNGradientKernel alloc] initWithDevice:_device];
    }
    _operation = _reduce;
}

NSString *GetReduceClass(ReduceType type, ReduceAxis axis)
{
    NSString *t = DescWithReduceType(type);
    NSString *a = DescWithReduceAxis(axis);
    return [NSString stringWithFormat:@"MPSNNReduce%@%@", a, t];
}

NSString *DescWithReduceType(ReduceType type)
{
    switch (type) {
        case ReduceTypeMax:
            return @"Max";
        case ReduceTypeMin:
            return @"Min";
        case ReduceTypeSum:
            return @"Sum";
        case ReduceTypeMean:
            return @"Mean";
        default:
            assert(0);
            return nil;
            break;
    }
}

NSString *DescWithReduceAxis(ReduceAxis axis)
{
    switch (axis) {
        case ReduceAxisRow:
            return @"Row";
        case ReduceAxisColumn:
            return @"Column";
        case ReduceAxisDepth:
            return @"FeatureChannels";
        default:
            assert(0);
            return nil;
            break;
    }
}

@end
