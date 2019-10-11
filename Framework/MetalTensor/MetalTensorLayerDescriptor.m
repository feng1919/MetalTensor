//
//  MetalTensorLayerDescriptor.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/25.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayerDescriptor.h"
#import "NSString+Extension.h"
#import "MIDataSource.h"
#import <MetalImage/MetalDevice.h>

Class DescriptorWithType(NSString *type)
{
    if ([type length] == 0) {
        // If the type were not specified
        return [MIConvolutionLayerDescriptor class];
    }
    if ([type isEqualToString:@"convolution"]) {
        return [MIConvolutionLayerDescriptor class];
    }
    if ([type isEqualToString:@"dense"]) {
        return [MIFullyConnectedLayerDescriptor class];
    }
    if ([type isEqualToString:@"softmax"]) {
        return [MISoftMaxLayerDescriptor class];
    }
    if ([type isEqualToString:@"pooling_average"]) {
        return [MIPoolingAverageLayerDescriptor class];
    }
    if ([type isEqualToString:@"pooling_max"]) {
        return [MIPoolingMaxLayerDescriptor class];
    }
    if ([type isEqualToString:@"reshape"]) {
        return [MIReshapeLayerDescriptor class];
    }
    if ([type isEqualToString:@"input"]) {
        return [MetalTensorInputLayerDescriptor class];
    }
    if ([type isEqualToString:@"output"]) {
        return [MetalTensorOutputLayerDescriptor class];
    }
    if ([type isEqualToString:@"concatenate"]) {
        return [MIConcatenateLayerDescriptor class];
    }
    if ([type isEqualToString:@"inverted_residual"]) {
        return [MIInvertedResidualModuleDescriptor class];
    }
    if ([type isEqualToString:@"arithmetic"]) {
        return [MIArithmeticLayerDescriptor class];
    }
    if ([type isEqualToString:@"neuron"]) {
        return [MetalTensorNeuronLayerDescriptor class];
    }
    if ([type isEqualToString:@"trans_conv"]) {
        return [MITransposeConvolutionLayerDescriptor class];
    }
    assert(0);
    return nil;
}

Class LayerWithType(NSString *type)
{
    if ([type isEqualToString:@"convolution"]) {
        return [MIConvolutionLayer class];
    }
    if ([type isEqualToString:@"dense"]) {
        return [MIFullyConnectedLayer class];
    }
    if ([type isEqualToString:@"softmax"]) {
        return [MISoftMaxLayer class];
    }
    if ([type isEqualToString:@"pooling_average"]) {
        return [MIPoolingAverageLayer class];
    }
    if ([type isEqualToString:@"pooling_max"]) {
        return [MIPoolingMaxLayer class];
    }
    if ([type isEqualToString:@"reshape"]) {
        return [MIReshapeLayer class];
    }
    if ([type isEqualToString:@"input"]) {
        return [MetalTensorInputLayer class];
    }
    if ([type isEqualToString:@"output"]) {
        return [MetalTensorOutputLayer class];
    }
    if ([type isEqualToString:@"concatenate"]) {
        return [MIConcatenateLayer class];
    }
    if ([type isEqualToString:@"inverted_residual"]) {
        return [MIInvertedResidualModule class];
    }
    if ([type isEqualToString:@"arithmetic"]) {
        return [MIArithmeticLayer class];
    }
    if ([type isEqualToString:@"neuron"]) {
        return [MetalTensorNeuronLayer class];
    }
    if ([type isEqualToString:@"trans_conv"]) {
        return [MITransposeConvolutionLayer class];
    }
    assert(0);
    return nil;
}

@implementation MetalTensorLayerDescriptor

- (instancetype)initWithDictionary:(NSDictionary *)dictionary {
    if (self = [super init]) {
        
        if (dictionary[@"inputs"]) {
            NSArray<NSString *> *inputsList = [dictionary[@"inputs"] nonEmptyComponentsSeparatedByString:@";"];
            _n_inputs = (int)inputsList.count;
            NSParameterAssert(_n_inputs > 0);
            _inputShapes = malloc(_n_inputs * sizeof(DataShape));
            for (int i = 0; i < inputsList.count; i++) {
                NSArray<NSString *> *inputInfo = [inputsList[i] nonEmptyComponentsSeparatedByString:@","];
                NSAssert(inputInfo.count == 3, @"Invliad input shape: '%@'", dictionary[@"inputs"]);
                _inputShapes[i].row = [inputInfo[0] intValue];
                _inputShapes[i].column = [inputInfo[1] intValue];
                _inputShapes[i].depth = [inputInfo[2] intValue];
            }
        }
        
        if (dictionary[@"output"]) {
            NSArray<NSString *> *outputList = [dictionary[@"output"] nonEmptyComponentsSeparatedByString:@","];
            NSAssert(outputList.count == 3, @"Invalid output shape: '%@'", dictionary[@"output"]);
            _outputShape.row = [outputList[0] intValue];
            _outputShape.column = [outputList[1] intValue];
            _outputShape.depth = [outputList[2] intValue];
        }
        
        _targets = [dictionary[@"targets"] nonEmptyComponentsSeparatedByString:@","];
        _targetIndices = [dictionary[@"indices"] nonEmptyComponentsSeparatedByString:@","];
        NSAssert(_targetIndices == nil || _targetIndices.count == _targets.count, @"If the indices were specified, the number of indices must be identical to the number of targets.");
        _type = dictionary[@"type"]?:@"convolution";
    }
    return self;
}

- (void)dealloc {
    if (_inputShapes) {
        free(_inputShapes);
    }
}

- (DataShape *)inputShapeRef {
    return _inputShapes;
}

- (DataShape *)outputShapeRef {
    return &_outputShape;
}

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MIConvolutionLayer

@implementation MIConvolutionLayerDescriptor

- (instancetype)initWithDictionary:(NSDictionary *)dictionary {
    if (self = [super initWithDictionary:dictionary]) {
        
        NSParameterAssert(dictionary[@"kernel"]);
        NSArray<NSString *> *kernelList = [dictionary[@"kernel"] nonEmptyComponentsSeparatedByString:@","];
        NSAssert(kernelList.count == 1 || kernelList.count == 2, @"Invalid kernel shape: '%@'", dictionary[@"kernel"]);
        _kernelShape.row = [kernelList[0] intValue];
        if (kernelList.count == 2) {
            _kernelShape.column = [kernelList[1] intValue];
        }
        else {
            _kernelShape.column = _kernelShape.row;
        }
        _kernelShape.depth = _inputShapes[0].depth;
        
        NSParameterAssert(dictionary[@"filters"]);
        _kernelShape.filters = [dictionary[@"filters"] intValue];
        
        if (dictionary[@"stride"]) {
            _kernelShape.stride = [dictionary[@"stride"] intValue];
            NSParameterAssert(_kernelShape.stride > 0);
        }
        else {
            _kernelShape.stride = 1;
        }
        
        if (dictionary[@"activation"]) {
            NSArray<NSString *> *neuronList = [dictionary[@"activation"] nonEmptyComponentsSeparatedByString:@","];
            _neuronType.neuron = NeuronTypeFromString(neuronList.firstObject);
            if (neuronList.count > 1) {
                _neuronType.a = [neuronList[1] floatValue];
            }
            if (neuronList.count > 2) {
                _neuronType.b = [neuronList[2] floatValue];
            }
        }
        
        if (dictionary[@"offset"]) {
            NSArray<NSString *> *offsetList = [dictionary[@"offset"] nonEmptyComponentsSeparatedByString:@","];
            NSAssert(offsetList.count == 3, @"Invliad offset number: '%@'", dictionary[@"offset"]);
            _offset.x = [offsetList[0] intValue];
            _offset.y = [offsetList[1] intValue];
            _offset.z = [offsetList[2] intValue];
        }
        else {
            _offset.x = _kernelShape.column % _kernelShape.stride;
            _offset.y = _kernelShape.row % _kernelShape.stride;
            _offset.z = 0;
        }
        
        _depthwise = [dictionary[@"depthwise"] boolValue];
        
        NSParameterAssert(dictionary[@"weight"]);
        _weight = dictionary[@"weight"];
        
        if (dictionary[@"weight_range"]) {
            NSArray<NSString *> *rangeList = [dictionary[@"weight_range"] nonEmptyComponentsSeparatedByString:@","];
            NSAssert(rangeList.count == 2, @"Invalid range number: '%@'", dictionary[@"weight_range"]);
            _weightRange.location = [rangeList[0] integerValue];
            _weightRange.length = [rangeList[1] integerValue];
        }
        else {
            _weightRange.location = NSNotFound;
            _weightRange.length = 0;
        }
    }
    return self;
}

- (KernelShape *)kernelShapeRef {
    return &_kernelShape;
}

- (NeuronType *)neuronTypeRef {
    return &_neuronType;
}

- (NSRange *)weightRangeRef {
    return &_weightRange;
}

@end

@implementation MIConvolutionLayer (layerDescriptorInit)

- (instancetype)initWithDescriptor:(MetalTensorLayerDescriptor *)descriptor {
    NSParameterAssert([descriptor isKindOfClass:[MIConvolutionLayerDescriptor class]]);
    if (self = [super initWithInputShape:[descriptor inputShapeRef] outputShape:[descriptor outputShapeRef]]) {
        MIConvolutionLayerDescriptor *convDesc = (MIConvolutionLayerDescriptor *)descriptor;
        
        NSString *weightPath = [[[NSBundle mainBundle] bundlePath] stringByAppendingPathComponent:convDesc.weight];
        if ([[NSFileManager defaultManager] fileExistsAtPath:weightPath]) {
            MICNNKernelDataSource *dataSource = MakeDataSource2(weightPath,
                                                                convDesc.kernelShapeRef,
                                                                convDesc.neuronTypeRef,
                                                                NO,
                                                                NULL);
            dataSource.range = convDesc.weightRange;
            self.dataSource = dataSource;
        }
        self.offset = convDesc.offset;
        self.edgeMode = MPSImageEdgeModeZero;
        [self setLabel:convDesc.name];
    }
    
    return self;
}

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MIReshapeLayer

@implementation MIReshapeLayerDescriptor

@end

@implementation MIReshapeLayer (layerDescriptorInit)

- (instancetype)initWithDescriptor:(MetalTensorLayerDescriptor *)descriptor {
    NSParameterAssert([descriptor isKindOfClass:[MIReshapeLayerDescriptor class]]);
    return [self initWithInputShape:[descriptor inputShapeRef] outputShape:[descriptor outputShapeRef]];
}

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MIConcatenateLayer

@implementation MIConcatenateLayerDescriptor

@end

@implementation MIConcatenateLayer (layerDescriptorInit)

- (instancetype)initWithDescriptor:(MetalTensorLayerDescriptor *)descriptor {
    NSParameterAssert([descriptor isKindOfClass:[MIConcatenateLayerDescriptor class]]);
    return [self initWithInputShapes:[descriptor inputShapes] size:[descriptor n_inputs]];
}

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MetalTensorInputLayer

@implementation MetalTensorInputLayerDescriptor

@end

@implementation MetalTensorInputLayer (layerDescriptorInit)

- (instancetype)initWithDescriptor:(MetalTensorLayerDescriptor *)descriptor {
    NSParameterAssert([descriptor isKindOfClass:[MetalTensorInputLayerDescriptor class]]);
    return [self initWithOutputShape:[descriptor outputShapeRef]];
}

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MetalTensorOutputLayer

@implementation MetalTensorOutputLayerDescriptor

@end

@implementation MetalTensorOutputLayer (layerDescriptorInit)

- (instancetype)initWithDescriptor:(MetalTensorLayerDescriptor *)descriptor {
    NSParameterAssert([descriptor isKindOfClass:[MetalTensorOutputLayerDescriptor class]]);
    return [self initWithOutputShape:[descriptor outputShapeRef]];
}

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MIInvertedResidualModule

@implementation MIInvertedResidualModuleDescriptor
@synthesize weightRanges = _weightRanges;

- (instancetype)initWithDictionary:(NSDictionary *)dictionary {
    if (self = [super initWithDictionary:dictionary]) {
        NSParameterAssert(dictionary[@"expansion"]);
        _expansion = [dictionary[@"expansion"] intValue];
        NSParameterAssert(_expansion>0);
        _stride = dictionary[@"stride"]?[dictionary[@"stride"] intValue]:1;
        NSParameterAssert(_stride>0);
        _filters = dictionary[@"filters"]?[dictionary[@"filters"] intValue]:_inputShapes[0].depth;
        NSParameterAssert(_filters>0);
        
        // It's shape should be specified, the first one or all three.
        // If only the first shape were specified, we may calculate the two others.
        NSParameterAssert(_n_inputs == 1 || _n_inputs == 3);
        if (_n_inputs == 1) {
            DataShape shape = _inputShapes[0];
            free(_inputShapes);
            _inputShapes = malloc(3*sizeof(DataShape));
            _inputShapes[0] = shape;
            int row = shape.row;
            int column = shape.column;
            int depth = shape.depth*_expansion;
            _inputShapes[1] = DataShapeMake(row, column, depth);
            row = (shape.row+_stride-1)/_stride;
            column = (shape.column+_stride-1)/_stride;
            NSParameterAssert(row > 0 && column > 0);
            _inputShapes[2] = DataShapeMake(row, column, depth);
        }
        
        // If the output were not specified, then we calcalate one.
        if (_outputShape.row == 0) {
            _outputShape.row = _inputShapes[2].row;
            _outputShape.column = _inputShapes[2].column;
            _outputShape.depth = _filters;
        }
        
        _kernelShapes = malloc(3*sizeof(KernelShape));
        MTLInt2 kernel;
        if (dictionary[@"kernel"]) {
            NSArray<NSString *> *kernelList = [dictionary[@"kernel"] nonEmptyComponentsSeparatedByString:@";"];
            NSAssert(kernelList.count == 1 || kernelList.count == 2, @"Invalid kernel number: '%@'", dictionary[@"kernel"]);
            kernel.y = [kernelList[0] intValue];
            if (kernelList.count == 2) {
                kernel.x = [kernelList[1] intValue];
            }
            else {
                kernel.x = kernel.y;
            }
        }
        else {
            kernel.x = kernel.y = 3;
        }
        int input_channels = _inputShapes[0].depth;
        input_channels = make_divisible_8(input_channels*_expansion);
        _kernelShapes[0] = KernelShapeMake(1, 1, _inputShapes[0].depth, input_channels, 1);
        _kernelShapes[1] = KernelShapeMake(kernel.y, kernel.x, input_channels, 1, _stride);
        _kernelShapes[2] = KernelShapeMake(1, 1, input_channels, _filters, 1);
        
        // If there were no neurons specified, we'll use [relu6, relu6, none] by default.
        _neuronTypes = malloc(3*sizeof(NeuronType));
        if (dictionary[@"neurons"]) {
            NSArray<NSString *> *neuronList = [dictionary[@"neurons"] nonEmptyComponentsSeparatedByString:@";"];
            NSParameterAssert(neuronList.count == 3);
            for (int i = 0; i < 3; i++) {
                NSArray<NSString *> *neuronInfo = [neuronList[i] nonEmptyComponentsSeparatedByString:@","];
                NSParameterAssert(neuronInfo.count == 3);
                _neuronTypes[i].neuron = NeuronTypeFromString(neuronInfo[0]);
                _neuronTypes[i].a = [neuronInfo[1] floatValue];
                _neuronTypes[i].b = [neuronInfo[2] floatValue];
            }
        }
        else {
            _neuronTypes[0].neuron = MPSCNNNeuronTypeReLUN;
            _neuronTypes[0].a = 0.0f;
            _neuronTypes[0].b = 6.0f;
            _neuronTypes[1].neuron = MPSCNNNeuronTypeReLUN;
            _neuronTypes[1].a = 0.0f;
            _neuronTypes[1].b = 6.0f;
            _neuronTypes[2].neuron = MPSCNNNeuronTypeNone;
            _neuronTypes[2].a = 0.0f;
            _neuronTypes[2].b = 0.0f;
        }
        
        NSParameterAssert(dictionary[@"weights"]);
        _weights = [dictionary[@"weights"] nonEmptyComponentsSeparatedByString:@","];
        if (_weights.count < 3) {
            NSMutableArray *array = [NSMutableArray arrayWithCapacity:3];
            [array addObject:_weights.firstObject];
            [array addObject:_weights.firstObject];
            [array addObject:_weights.firstObject];
            _weights = [NSArray arrayWithArray:array];
        }
        
        if (dictionary[@"offset"]) {
            NSArray<NSString *> *offsetList = [dictionary[@"offset"] nonEmptyComponentsSeparatedByString:@","];
            NSAssert(offsetList.count == 3, @"Invliad offset number: '%@'", dictionary[@"offset"]);
            _offset.x = [offsetList[0] intValue];
            _offset.y = [offsetList[1] intValue];
            _offset.z = [offsetList[2] intValue];
        }
        else {
            _offset.x = kernel.x % _stride;
            _offset.y = kernel.y % _stride;
            _offset.z = 0;
        }
        
        if (dictionary[@"weight_ranges"]) {
            NSArray<NSString *> *rangeList = [dictionary[@"weight_ranges"] nonEmptyComponentsSeparatedByString:@";"];
            NSAssert(rangeList.count == 3, @"Invlid weight range number: '%@'", dictionary[@"weight_ranges"]);
            _weightRanges = malloc(3*sizeof(NSRange));
            for (int i = 0; i < 3; i++) {
                NSArray<NSString *> *rangeComponents = [rangeList[i] nonEmptyComponentsSeparatedByString:@","];
                _weightRanges[i].location = [rangeComponents[0] integerValue];
                _weightRanges[i].length = [rangeComponents[1] integerValue];
            }
        }
        else {
            _weightRanges = NULL;
        }
    }
    return self;
}

- (void)dealloc {
    free(_kernelShapes);
    free(_neuronTypes);
    
    if (_weightRanges) {
        free(_weightRanges);
    }
}

- (NSRange *)weightRanges {
    return _weightRanges;
}

@end


@implementation MIInvertedResidualModule (layerDescriptorInit)

- (instancetype)initWithDescriptor:(MetalTensorLayerDescriptor *)descriptor {
    NSParameterAssert([descriptor isKindOfClass:[MIInvertedResidualModuleDescriptor class]]);
    MIInvertedResidualModuleDescriptor *irmDesc = (MIInvertedResidualModuleDescriptor *)descriptor;
    DataShape *inputShapes = [irmDesc inputShapes];
    
    MIInvertedResidualModule *invertedResidualModule = [self initWithInputShape:&inputShapes[0] outputShape:[irmDesc outputShapeRef] dwInputShape:&inputShapes[1] dwOutputShape:&inputShapes[2]];
    
    KernelShape *kernels = [irmDesc kernelShapes];
    NeuronType *neurons = [irmDesc neuronTypes];
    NSRange *weightRanges = [irmDesc weightRanges];
    
    NSString *weightPath = [[NSBundle mainBundle] pathForResource:irmDesc.weights[0] ofType:@"bin"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:weightPath]) {
        MICNNKernelDataSource *dataSource = MakeDataSource1(irmDesc.weights[0],
                                                             &kernels[0],
                                                             &neurons[0],
                                                             NO);
        if (weightRanges) {
            dataSource.range = weightRanges[0];
        }
        [invertedResidualModule setExpandDataSource:dataSource];
    }
    
    weightPath = [[NSBundle mainBundle] pathForResource:irmDesc.weights[1] ofType:@"bin"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:weightPath]) {
        MICNNKernelDataSource *dataSource = MakeDataSource1(irmDesc.weights[1],
                                                             &kernels[1],
                                                             &neurons[1],
                                                             YES);
        if (weightRanges) {
            dataSource.range = weightRanges[1];
        }
        [invertedResidualModule setDepthWiseDataSource:dataSource];
    }
    
    weightPath = [[NSBundle mainBundle] pathForResource:irmDesc.weights[2] ofType:@"bin"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:weightPath]) {
        MICNNKernelDataSource *dataSource = MakeDataSource1(irmDesc.weights[2],
                                                            &kernels[2],
                                                            &neurons[2],
                                                            NO);
        if (weightRanges) {
            dataSource.range = weightRanges[2];
        }
        [invertedResidualModule setProjectDataSource:dataSource];
    }
    invertedResidualModule.label = descriptor.name;
    [[invertedResidualModule depthWiseComponent] setOffset:irmDesc.offset];
    return invertedResidualModule;
}

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MISoftMaxLayer

@implementation MISoftMaxLayerDescriptor

@end

@implementation MISoftMaxLayer (layerDescriptorInit)

- (instancetype)initWithDescriptor:(MetalTensorLayerDescriptor *)descriptor {
    return [self initWithInputShape:&descriptor.inputShapes[0] outputShape:[descriptor outputShapeRef]];
}

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MIFullyConnectedLayer

@implementation MIFullyConnectedLayerDescriptor

- (instancetype)initWithDictionary:(NSDictionary *)dictionary {
    if (self = [super initWithDictionary:dictionary]) {
        
        NSParameterAssert(dictionary[@"kernel"]);
        NSArray<NSString *> *kernelList = [dictionary[@"kernel"] nonEmptyComponentsSeparatedByString:@","];
        NSAssert(kernelList.count == 1 || kernelList.count == 2, @"Invalid kernel shape: '%@'", dictionary[@"kernel"]);
        _kernelShape.row = [kernelList[0] intValue];
        if (kernelList.count == 2) {
            _kernelShape.column = [kernelList[1] intValue];
        }
        else {
            _kernelShape.column = _kernelShape.row;
        }
        _kernelShape.depth = _inputShapes[0].depth;
        
        NSParameterAssert(dictionary[@"filters"]);
        _kernelShape.filters = [dictionary[@"filters"] intValue];
        
        if (dictionary[@"stride"]) {
            _kernelShape.stride = [dictionary[@"stride"] intValue];
            NSParameterAssert(_kernelShape.stride > 0);
        }
        else {
            _kernelShape.stride = 1;
        }
        
        if (dictionary[@"activation"]) {
            NSArray<NSString *> *neuronList = [dictionary[@"activation"] nonEmptyComponentsSeparatedByString:@","];
            _neuronType.neuron = NeuronTypeFromString(neuronList.firstObject);
            if (neuronList.count > 1) {
                _neuronType.a = [neuronList[1] floatValue];
            }
            if (neuronList.count > 2) {
                _neuronType.b = [neuronList[2] floatValue];
            }
            if (neuronList.count > 3) {
                _neuronType.c = [neuronList[3] floatValue];
            }
        }
        else {
            _neuronType.neuron = MPSCNNNeuronTypeNone;
            _neuronType.a = 0.0f;
            _neuronType.b = 0.0f;
            _neuronType.c = 0.0f;
        }
        
        NSParameterAssert(dictionary[@"weight"]);
        _weight = dictionary[@"weight"];
        
        if (dictionary[@"weight_range"]) {
            NSArray<NSString *> *rangeList = [dictionary[@"weight_range"] nonEmptyComponentsSeparatedByString:@","];
            NSAssert(rangeList.count == 2, @"Invalid range number: '%@'", dictionary[@"weight_range"]);
            _weightRange.location = [rangeList[0] integerValue];
            _weightRange.length = [rangeList[1] integerValue];
        }
        else {
            _weightRange.location = NSNotFound;
            _weightRange.length = 0;
        }
    }
    return self;
}

- (KernelShape *)kernelShapeRef {
    return &_kernelShape;
}

- (NeuronType *)neuronTypeRef {
    return &_neuronType;
}

- (NSRange *)weightRangeRef {
    return &_weightRange;
}

@end

@implementation MIFullyConnectedLayer (layerDescriptorInit)

- (instancetype)initWithDescriptor:(MetalTensorLayerDescriptor *)descriptor {
    NSParameterAssert([descriptor isKindOfClass:[MIFullyConnectedLayerDescriptor class]]);
    if (self = [super initWithInputShape:[descriptor inputShapeRef] outputShape:[descriptor outputShapeRef]]) {
        MIFullyConnectedLayerDescriptor *denseDesc = (MIFullyConnectedLayerDescriptor *)descriptor;
        
        NSString *weightPath = [[NSBundle mainBundle] pathForResource:denseDesc.weight ofType:@"bin"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:weightPath]) {
            MICNNKernelDataSource *dataSource = MakeDataSource1(denseDesc.weight,
                                                                denseDesc.kernelShapeRef,
                                                                denseDesc.neuronTypeRef,
                                                                NO);
            dataSource.range = denseDesc.weightRange;
            self.dataSource = dataSource;
        }
        [self setLabel:denseDesc.name];
    }
    
    return self;
}

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MIPoolingAverageLayer

@implementation MIPoolingAverageLayerDescriptor

- (instancetype)initWithDictionary:(NSDictionary *)dictionary {
    if (self = [super initWithDictionary:dictionary]) {
        
        NSParameterAssert(dictionary[@"kernel"]);
        NSArray<NSString *> *kernelList = [dictionary[@"kernel"] nonEmptyComponentsSeparatedByString:@","];
        NSAssert(kernelList.count == 1 || kernelList.count == 2, @"Invalid kernel shape: '%@'", dictionary[@"kernel"]);
        _kernelShape.row = [kernelList[0] intValue];
        if (kernelList.count == 2) {
            _kernelShape.column = [kernelList[1] intValue];
        }
        else {
            _kernelShape.column = _kernelShape.row;
        }
        _kernelShape.depth = _inputShapes[0].depth;
        _kernelShape.filters = _inputShapes[0].depth;
        
        if (dictionary[@"stride"]) {
            _kernelShape.stride = [dictionary[@"stride"] intValue];
            NSParameterAssert(_kernelShape.stride > 0);
        }
        else {
            _kernelShape.stride = 1;
        }
        
        if (dictionary[@"offset"]) {
            NSArray<NSString *> *offsetList = [dictionary[@"offset"] nonEmptyComponentsSeparatedByString:@","];
            NSAssert(offsetList.count == 3, @"Invliad offset number: '%@'", dictionary[@"offset"]);
            _offset.x = [offsetList[0] intValue];
            _offset.y = [offsetList[1] intValue];
            _offset.z = [offsetList[2] intValue];
        }
        else {
            _offset.x = _kernelShape.column % _kernelShape.stride;
            _offset.y = _kernelShape.row % _kernelShape.stride;
            _offset.z = 0;
        }
    }
    return self;
}

@end

@implementation MIPoolingAverageLayer (layerDescriptorInit)

- (instancetype)initWithDescriptor:(MetalTensorLayerDescriptor *)descriptor {
    NSParameterAssert([descriptor isKindOfClass:[MIPoolingAverageLayerDescriptor class]]);
    if (self = [super initWithInputShape:&descriptor.inputShapes[0] outputShape:[descriptor outputShapeRef]]) {
        MIPoolingAverageLayerDescriptor *poolingDesc = (MIPoolingAverageLayerDescriptor *)descriptor;
        self.kernelWidth = poolingDesc.kernelShape.column;
        self.kernelHeight = poolingDesc.kernelShape.row;
        [self setOffset:poolingDesc.offset];
        self.strideInPixelsX = poolingDesc.kernelShape.stride;
        self.strideInPixelsY = poolingDesc.kernelShape.stride;
    }
    return self;
}

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MIPoolingMaxLayer

@implementation MIPoolingMaxLayerDescriptor

- (instancetype)initWithDictionary:(NSDictionary *)dictionary {
    if (self = [super initWithDictionary:dictionary]) {
        
        NSParameterAssert(dictionary[@"kernel"]);
        NSArray<NSString *> *kernelList = [dictionary[@"kernel"] nonEmptyComponentsSeparatedByString:@","];
        NSAssert(kernelList.count == 1 || kernelList.count == 2, @"Invalid kernel shape: '%@'", dictionary[@"kernel"]);
        _kernelShape.row = [kernelList[0] intValue];
        if (kernelList.count == 2) {
            _kernelShape.column = [kernelList[1] intValue];
        }
        else {
            _kernelShape.column = _kernelShape.row;
        }
        _kernelShape.depth = _inputShapes[0].depth;
        _kernelShape.filters = _inputShapes[0].depth;
        
        if (dictionary[@"stride"]) {
            _kernelShape.stride = [dictionary[@"stride"] intValue];
            NSParameterAssert(_kernelShape.stride > 0);
        }
        else {
            _kernelShape.stride = 1;
        }
        
        if (dictionary[@"offset"]) {
            NSArray<NSString *> *offsetList = [dictionary[@"offset"] nonEmptyComponentsSeparatedByString:@","];
            NSAssert(offsetList.count == 3, @"Invliad offset number: '%@'", dictionary[@"offset"]);
            _offset.x = [offsetList[0] intValue];
            _offset.y = [offsetList[1] intValue];
            _offset.z = [offsetList[2] intValue];
        }
        else {
            _offset.x = _kernelShape.column % _kernelShape.stride;
            _offset.y = _kernelShape.row % _kernelShape.stride;
            _offset.z = 0;
        }
    }
    return self;
}

@end

@implementation MIPoolingMaxLayer (layerDescriptorInit)

- (instancetype)initWithDescriptor:(MIPoolingMaxLayerDescriptor *)descriptor {
    NSParameterAssert([descriptor isKindOfClass:[MIPoolingMaxLayerDescriptor class]]);
    if (self = [super initWithInputShape:&descriptor.inputShapes[0] outputShape:[descriptor outputShapeRef]]) {
        MIPoolingMaxLayerDescriptor *poolingDesc = (MIPoolingMaxLayerDescriptor *)descriptor;
        self.kernelWidth = poolingDesc.kernelShape.column;
        self.kernelHeight = poolingDesc.kernelShape.row;
        [self setOffset:poolingDesc.offset];
        self.strideInPixelsX = poolingDesc.kernelShape.stride;
        self.strideInPixelsY = poolingDesc.kernelShape.stride;
    }
    return self;
}

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MIArithmeticLayer

@implementation MIArithmeticLayerDescriptor

- (instancetype)initWithDictionary:(NSDictionary *)dictionary {
    if (self = [super initWithDictionary:dictionary]) {
        
        NSParameterAssert(dictionary[@"data_shape"]);
        NSArray<NSString *> *shapeList = [dictionary[@"data_shape"] nonEmptyComponentsSeparatedByString:@","];
        NSAssert(shapeList.count == 3, @"Invalid data shape: '%@'", dictionary[@"data_shape"]);
        _dataShape.row = [shapeList[0] intValue];
        _dataShape.column = [shapeList[1] intValue];
        _dataShape.depth = [shapeList[2] intValue];
        
        NSParameterAssert(dictionary[@"arithmetic"]);
        _arithmetic = dictionary[@"arithmetic"];
        
        _channelOffset = [dictionary[@"channel_offset"] integerValue];
        _secondaryImage = dictionary[@"secondary_image"];
    }
    return self;
}

- (DataShape *)dataShapeRef {
    return &_dataShape;
}

@end

@implementation MIArithmeticLayer (layerDescriptorInit)

- (instancetype)initWithDescriptor:(MIArithmeticLayerDescriptor *)descriptor {
    NSParameterAssert([descriptor isKindOfClass:[MIArithmeticLayerDescriptor class]]);
    DataShape *dataShape = [descriptor dataShapeRef];
    DataShape *inputShapes[2] = {dataShape, dataShape};
    if (self = [super initWithInputShapes1:inputShapes size:2 outputShape:dataShape]) {
        self->dataShape = *dataShape;
        
        Class arithmeticClass = [MIArithmeticLayer arithmeticWithType:descriptor.arithmetic];
        self->arithmetic = [[arithmeticClass alloc] initWithDevice:[MetalDevice sharedMTLDevice]];
        self->arithmetic.primaryScale = 1.0f;
        self->arithmetic.bias = 0.0f;
        self->arithmetic.primaryStrideInPixelsX = 1;
        self->arithmetic.primaryStrideInPixelsY = 1;
        self->arithmetic.primaryStrideInFeatureChannels = 1;
        self->arithmetic.secondaryScale = 1.0f;
        self->arithmetic.secondaryStrideInPixelsX = 1;
        self->arithmetic.secondaryStrideInPixelsY = 1;
        self->arithmetic.secondaryStrideInFeatureChannels = 1;
        self->arithmetic.destinationFeatureChannelOffset = descriptor.channelOffset;
        
        if (descriptor.secondaryImage) {
            UIImage *secondaryImage = [UIImage imageNamed:descriptor.secondaryImage];
            if (!secondaryImage) {
                secondaryImage = [UIImage imageWithContentsOfFile:descriptor.secondaryImage];
            }
            NSAssert(secondaryImage, @"Failed to load the secondary image: %@", descriptor.secondaryImage);
            self.secondaryImage = [[MIMPSImage alloc] initWithImage:secondaryImage normalized:YES];
        }
    }
    return self;
}

+ (Class)arithmeticWithType:(NSString *)arithmetic {
    if ([arithmetic isEqualToString:@"addition"]) {
        return [MPSCNNAdd class];
    }
    if ([arithmetic isEqualToString:@"divide"]) {
        return [MPSCNNDivide class];
    }
    if ([arithmetic isEqualToString:@"subtract"]) {
        return [MPSCNNSubtract class];
    }
    if ([arithmetic isEqualToString:@"multiply"]) {
        return [MPSCNNMultiply class];
    }
    assert(0);
    return nil;
}

@end


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MetalTensorNeuronLayer

@implementation MetalTensorNeuronLayerDescriptor

- (instancetype)initWithDictionary:(NSDictionary *)dictionary {
    if (self = [super initWithDictionary:dictionary]) {
        
        NSParameterAssert(dictionary[@"data_shape"]);
        NSArray<NSString *> *shapeInfo = [dictionary[@"data_shape"] nonEmptyComponentsSeparatedByString:@","];
        NSAssert(shapeInfo.count == 3, @"Invalid data shape count");
        _dataShape.row = [shapeInfo[0] intValue];
        _dataShape.column = [shapeInfo[1] intValue];
        _dataShape.depth = [shapeInfo[2] intValue];
        
        NSParameterAssert(dictionary[@"activation"]);
        NSArray<NSString *> *neuronInfo = [dictionary[@"activation"] nonEmptyComponentsSeparatedByString:@","];
        _neuronType.neuron = NeuronTypeFromString(neuronInfo[0]);
        if (neuronInfo.count > 1) {
            _neuronType.a = [neuronInfo[1] floatValue];
        }
        if (neuronInfo.count > 2) {
            _neuronType.b = [neuronInfo[2] floatValue];
        }
        if (neuronInfo.count > 3) {
            _neuronType.c = [neuronInfo[3] floatValue];
        }
    }
    return self;
}

- (DataShape *)dataShapeRef {
    return &_dataShape;
}

- (NeuronType *)neuronTypeRef {
    return &_neuronType;
}

@end

@implementation MetalTensorNeuronLayer (layerDescriptorInit)

- (instancetype)initWithDescriptor:(MetalTensorNeuronLayerDescriptor *)descriptor {
    NSParameterAssert([descriptor isKindOfClass:[MetalTensorNeuronLayerDescriptor class]]);
    return [self initWithDataShape:descriptor.dataShapeRef neuronType:descriptor.neuronType];
}

@end


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MITransposeConvolutionLayer

@implementation MITransposeConvolutionLayerDescriptor

- (instancetype)initWithDictionary:(NSDictionary *)dictionary {
    if (self = [super initWithDictionary:dictionary]) {
        
        NSParameterAssert(dictionary[@"kernel"]);
        NSArray<NSString *> *kernelList = [dictionary[@"kernel"] nonEmptyComponentsSeparatedByString:@","];
        NSAssert(kernelList.count == 1 || kernelList.count == 2, @"Invalid kernel shape: '%@'", dictionary[@"kernel"]);
        _kernelShape.row = [kernelList[0] intValue];
        if (kernelList.count == 2) {
            _kernelShape.column = [kernelList[1] intValue];
        }
        else {
            _kernelShape.column = _kernelShape.row;
        }
        _kernelShape.depth = _inputShapes[0].depth;
        
        NSParameterAssert(dictionary[@"filters"]);
        _kernelShape.filters = [dictionary[@"filters"] intValue];
        
        if (dictionary[@"stride"]) {
            _kernelShape.stride = [dictionary[@"stride"] intValue];
            NSParameterAssert(_kernelShape.stride > 0);
        }
        else {
            _kernelShape.stride = 1;
        }
        
        if (dictionary[@"activation"]) {
            NSArray<NSString *> *neuronList = [dictionary[@"activation"] nonEmptyComponentsSeparatedByString:@","];
            _neuronType.neuron = NeuronTypeFromString(neuronList.firstObject);
            if (neuronList.count > 1) {
                _neuronType.a = [neuronList[1] floatValue];
            }
            if (neuronList.count > 2) {
                _neuronType.b = [neuronList[2] floatValue];
            }
        }
        
        if (dictionary[@"offset"]) {
            NSArray<NSString *> *offsetList = [dictionary[@"offset"] nonEmptyComponentsSeparatedByString:@","];
            NSAssert(offsetList.count == 3, @"Invliad offset number: '%@'", dictionary[@"offset"]);
            _offset.x = [offsetList[0] intValue];
            _offset.y = [offsetList[1] intValue];
            _offset.z = [offsetList[2] intValue];
        }
        
        if (dictionary[@"kernel_offset"]) {
            NSArray<NSString *> *offsetList = [dictionary[@"kernel_offset"] nonEmptyComponentsSeparatedByString:@","];
            NSAssert(offsetList.count == 2, @"Invliad kernel offset number: '%@'", dictionary[@"kernel_offset"]);
            _kernelOffset.x = [offsetList[0] intValue];
            _kernelOffset.y = [offsetList[1] intValue];
        }
        else {
            _kernelOffset.x = -1 * (_kernelShape.column % _kernelShape.stride);
            _kernelOffset.y = -1 * (_kernelShape.row % _kernelShape.stride);
        }
        
        NSParameterAssert(dictionary[@"weight"]);
        _weight = dictionary[@"weight"];
        
        if (dictionary[@"weight_range"]) {
            NSArray<NSString *> *rangeList = [dictionary[@"weight_range"] nonEmptyComponentsSeparatedByString:@","];
            NSAssert(rangeList.count == 2, @"Invalid range number: '%@'", dictionary[@"weight_range"]);
            _weightRange.location = [rangeList[0] integerValue];
            _weightRange.length = [rangeList[1] integerValue];
        }
        else {
            _weightRange.location = NSNotFound;
            _weightRange.length = 0;
        }
    }
    return self;
}

- (KernelShape *)kernelShapeRef {
    return &_kernelShape;
}

- (NeuronType *)neuronTypeRef {
    return &_neuronType;
}

- (NSRange *)weightRangeRef {
    return &_weightRange;
}

@end

@implementation MITransposeConvolutionLayer (layerDescriptorInit)

- (instancetype)initWithDescriptor:(MetalTensorLayerDescriptor *)descriptor {
    NSParameterAssert([descriptor isKindOfClass:[MITransposeConvolutionLayerDescriptor class]]);
    if (self = [super initWithInputShape:[descriptor inputShapeRef] outputShape:[descriptor outputShapeRef]]) {
        MITransposeConvolutionLayerDescriptor *convDesc = (MITransposeConvolutionLayerDescriptor *)descriptor;
        
        NSString *weightPath = [[NSBundle mainBundle] pathForResource:convDesc.weight ofType:@"bin"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:weightPath]) {
            MICNNKernelDataSource *dataSource = MakeDataSource1(convDesc.weight,
                                                                convDesc.kernelShapeRef,
                                                                convDesc.neuronTypeRef,
                                                                NO);
            dataSource.range = convDesc.weightRange;
            self.dataSource = dataSource;
        }
        self.offset = convDesc.offset;
        self.kernelOffset = convDesc.kernelOffset;
        self.edgeMode = MPSImageEdgeModeZero;
        [self setLabel:convDesc.name];
    }
    
    return self;
}

@end