//
//  MIDataSource.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/22.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import "MPSImage+Extension.h"
#import "metal_tensor_structures.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIDataSource : NSObject {
    
@protected
    NSString *_filePath;
    NSData *_data;
    NSString *_label;
    NSRange _range;
    BOOL _purgeEnable;
}

@property (nonatomic, strong) NSString *filePath;
@property (nonatomic, assign) NSRange range;
@property (nonatomic, strong, nullable) NSString *label;// used for debug

- (instancetype)initWithContentOfFile:(NSString *)filePath;
- (NSData *)data;

@end

@interface MIBNParametersDataSource : MIDataSource <MPSCNNBatchNormalizationDataSource, NSCopying>

@end

@interface MICNNKernelDataSource : MIDataSource <NSCopying, MPSCNNConvolutionDataSource> {
    
@protected
    KernelShape _kernel;
    NeuronType _neuron;// default ReLU6
    MPSCNNConvolutionDescriptor *_descriptor;
    MIBNParametersDataSource *_bn;
    BOOL _depthWise;
}

@property (nonatomic, assign) KernelShape kernel;
@property (nonatomic, assign) NeuronType neuron;
@property (nonatomic, strong) MIBNParametersDataSource *bn;
@property (nonatomic, assign) BOOL depthWise;
@property (nonatomic, assign) BOOL removeBias;

/*
 *  Transpose input channels and output channels of weights.
 *  Default by NO.
 */
@property (nonatomic, assign) BOOL transposeIO;

/*
 *  It is different weights treatments between tensorflow and Metal in
 *  transpose convolution and backward gradients convolution.
 *  Default by NO.
 */
@property (nonatomic, assign) BOOL rotateSpatial180;

- (instancetype)initWithData:(NSData *)weights
                      kernel:(KernelShape *)kernel
                      neuron:(NeuronType *)neuron
                   depthWise:(BOOL)depthWise;
- (instancetype)initWithContentOfFile:(NSString *)filePath
                               kernel:(KernelShape *)kernel
                               neuron:(NeuronType *)neuron;
- (instancetype)initWithContentOfFile:(NSString *)filePath
                               kernel:(KernelShape *)kernel
                               neuron:(NeuronType *)neuron
                            depthWise:(BOOL)depthWise;

@end

void transpose_input_output_channels(float32_t *buffer, KernelShape *kernel);
void rotate_spatial_180(float32_t *buffer, KernelShape *kernel);

MICNNKernelDataSource *MakeDataSource(NSString *module_name, KernelShape *k, NeuronType *n);
MICNNKernelDataSource *MakeDataSource1(NSString *module_name, KernelShape *k, NeuronType *n, BOOL isDepthWise);
MICNNKernelDataSource *MakeDataSource2(NSString *weight_path, KernelShape *k, NeuronType *n, BOOL isDepthWise, NSRange * _Nullable range);

NS_ASSUME_NONNULL_END

/**
 *  PREPROCESS: SWap B & R Channels
 *  MetalImage uses BGBA 8-bit unsigned int as default image format
 *  MPSImage uses RGBA 16-bit float as default image format
 *  This layer performs a static convolution on the input MPSImage,
 *  to swaps R & B channels.
 *
 **/

