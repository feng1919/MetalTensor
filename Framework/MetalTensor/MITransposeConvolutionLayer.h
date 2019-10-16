//
//  MITransposeConvolutionLayer.h
//  MetalTensorDemo
//
//  Created by Feng Stone on 2019/9/24.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"
#import "MIDataSource.h"

NS_ASSUME_NONNULL_BEGIN

@interface MITransposeConvolutionLayer : MetalTensorLayer <MetalTensorWeights>

@property (nonatomic, assign) KernelShape kernel;
@property (nonatomic, assign) NeuronType neuron;
@property (nonatomic, assign) MTPaddingMode padding;
@property (nonatomic, assign) BOOL depthWise;
@property (nonatomic, assign) MPSImageEdgeMode edgeMode;
@property (nonatomic, assign) MPSOffset offset;
@property (nonatomic, assign) MTLInt2 kernelOffset;
@property (nonatomic, strong) MICNNKernelDataSource *dataSource;

@end

NS_ASSUME_NONNULL_END
