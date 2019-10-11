//
//  MetalTensorLayerDescriptor.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/25.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MPSImage+Extension.h"
#import "MIConvolutionLayer.h"
#import "MIReshapeLayer.h"
#import "MIConcatenateLayer.h"
#import "MetalTensorInputLayer.h"
#import "MetalTensorOutputLayer.h"
#import "MIInvertedResidualModule.h"
#import "MISoftMaxLayer.h"
#import "MIFullyConnectedLayer.h"
#import "MIPoolingAverageLayer.h"
#import "MIPoolingMaxLayer.h"
#import "MIArithmeticLayer.h"
#import "MetalTensorNeuronLayer.h"
#import "MITransposeConvolutionLayer.h"

NS_ASSUME_NONNULL_BEGIN

Class DescriptorWithType(NSString *type);
Class LayerWithType(NSString *type);

@interface MetalTensorLayerDescriptor : NSObject {
    
@protected
    NSString *_name;
    NSString *_type;
    NSArray<NSString *> *_targets;
    NSArray<NSString *> *_targetIndices;
    DataShape _outputShape;
    DataShape *_inputShapes;
    int _n_inputs;
}

@property (nonatomic, retain, nullable) NSString *name;
@property (nonatomic, retain, readonly, nullable) NSString *type;
@property (nonatomic, retain, readonly, nullable) NSArray<NSString *> *targets;
@property (nonatomic, retain, readonly, nullable) NSArray<NSString *> *targetIndices;
@property (nonatomic, readonly) DataShape outputShape;
@property (nonatomic, readonly) DataShape *inputShapes;
@property (nonatomic, readonly) int n_inputs;

- (instancetype)initWithDictionary:(NSDictionary *)dictionary;

- (DataShape *)inputShapeRef;
- (DataShape *)outputShapeRef;

@end

@protocol MetalTensorLayerDescriptorInitialize <NSObject>

@required
- (instancetype)initWithDescriptor:(MetalTensorLayerDescriptor *)descriptor;

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MIConvolutionLayer

@interface MIConvolutionLayerDescriptor : MetalTensorLayerDescriptor 

@property (nonatomic, readonly) KernelShape kernelShape;
@property (nonatomic, readonly) NeuronType neuronType;
@property (nonatomic, readonly) MPSOffset offset;

@property (nonatomic, readonly) BOOL depthwise;
@property (nonatomic, retain, readonly, nullable) NSString *weight;
@property (nonatomic, assign, readonly) NSRange weightRange;

- (KernelShape *)kernelShapeRef;
- (NeuronType *)neuronTypeRef;
- (NSRange *)weightRangeRef;

@end

@interface MIConvolutionLayer (layerDescriptorInit) <MetalTensorLayerDescriptorInitialize>

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MIReshapeLayer

@interface MIReshapeLayerDescriptor : MetalTensorLayerDescriptor

@end

@interface MIReshapeLayer (layerDescriptorInit) <MetalTensorLayerDescriptorInitialize>

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MIConcatenateLayer

@interface MIConcatenateLayerDescriptor : MetalTensorLayerDescriptor

@end

@interface MIConcatenateLayer (layerDescriptorInit) <MetalTensorLayerDescriptorInitialize>

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MetalTensorInputLayer

@interface MetalTensorInputLayerDescriptor : MetalTensorLayerDescriptor

@end

@interface MetalTensorInputLayer (layerDescriptorInit) <MetalTensorLayerDescriptorInitialize>

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MetalTensorOutputLayer

@interface MetalTensorOutputLayerDescriptor : MetalTensorLayerDescriptor

@end

@interface MetalTensorOutputLayer (layerDescriptorInit) <MetalTensorLayerDescriptorInitialize>

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MIInvertedResidualModule

@interface MIInvertedResidualModuleDescriptor : MetalTensorLayerDescriptor

@property (nonatomic, readonly) int expansion;
@property (nonatomic, readonly) int stride;
@property (nonatomic, readonly) int filters;

@property (nonatomic, readonly) KernelShape *kernelShapes;
@property (nonatomic, readonly) NeuronType *neuronTypes; // default: [relu6, relu6, none]
@property (nonatomic, readonly) MPSOffset offset;
@property (nonatomic, readonly, strong) NSArray<NSString *> *weights;
@property (nonatomic, readonly) NSRange *weightRanges;

@end

@interface MIInvertedResidualModule (layerDescriptorInit) <MetalTensorLayerDescriptorInitialize>

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MISoftMaxLayer

@interface MISoftMaxLayerDescriptor : MetalTensorLayerDescriptor


@end

@interface MISoftMaxLayer (layerDescriptorInit) <MetalTensorLayerDescriptorInitialize>

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MIFullyConnectedLayer

@interface MIFullyConnectedLayerDescriptor : MetalTensorLayerDescriptor

@property (nonatomic, readonly) KernelShape kernelShape;
@property (nonatomic, readonly) NeuronType neuronType;
@property (nonatomic, retain, readonly, nullable) NSString *weight;
@property (nonatomic, assign, readonly) NSRange weightRange;

- (KernelShape *)kernelShapeRef;
- (NeuronType *)neuronTypeRef;
- (NSRange *)weightRangeRef;

@end

@interface MIFullyConnectedLayer (layerDescriptorInit) <MetalTensorLayerDescriptorInitialize>

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MIPoolingAverageLayer

@interface MIPoolingAverageLayerDescriptor : MetalTensorLayerDescriptor

@property (nonatomic, readonly) KernelShape kernelShape;
@property (nonatomic, readonly) MPSOffset offset;

@end

@interface MIPoolingAverageLayer (layerDescriptorInit) <MetalTensorLayerDescriptorInitialize>

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MIPoolingMaxLayer

@interface MIPoolingMaxLayerDescriptor : MetalTensorLayerDescriptor

@property (nonatomic, readonly) KernelShape kernelShape;
@property (nonatomic, readonly) MPSOffset offset;

@end

@interface MIPoolingMaxLayer (layerDescriptorInit) <MetalTensorLayerDescriptorInitialize>

@end


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MIArithmeticLayer

@interface MIArithmeticLayerDescriptor : MetalTensorLayerDescriptor

@property (nonatomic, readonly) DataShape dataShape;
@property (nonatomic, readonly) NSInteger channelOffset;
@property (nonatomic, strong, readonly) NSString *arithmetic;
@property (nonatomic, strong, readonly) NSString *secondaryImage;
@property (nonatomic, readonly) BOOL normalized;

- (DataShape *)dataShapeRef;

@end

@interface MIArithmeticLayer (layerDescriptorInit) <MetalTensorLayerDescriptorInitialize>

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MetalTensorNeuronLayer

@interface MetalTensorNeuronLayerDescriptor : MetalTensorLayerDescriptor

@property (nonatomic, readonly) DataShape dataShape;
@property (nonatomic, readonly) NeuronType neuronType;

- (DataShape *)dataShapeRef;
- (NeuronType *)neuronTypeRef;

@end

@interface MetalTensorNeuronLayer (layerDescriptorInit) <MetalTensorLayerDescriptorInitialize>

@end

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark - MITransposeConvolutionLayer

@interface MITransposeConvolutionLayerDescriptor : MetalTensorLayerDescriptor

@property (nonatomic, readonly) KernelShape kernelShape;
@property (nonatomic, readonly) NeuronType neuronType;
@property (nonatomic, readonly) MPSOffset offset;
@property (nonatomic, readonly) MTLInt2 kernelOffset;

@property (nonatomic, retain, readonly, nullable) NSString *weight;
@property (nonatomic, assign, readonly) NSRange weightRange;

- (KernelShape *)kernelShapeRef;
- (NeuronType *)neuronTypeRef;
- (NSRange *)weightRangeRef;

@end

@interface MITransposeConvolutionLayer (layerDescriptorInit) <MetalTensorLayerDescriptorInitialize>

@end

NS_ASSUME_NONNULL_END