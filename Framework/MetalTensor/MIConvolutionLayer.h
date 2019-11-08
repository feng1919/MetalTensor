//
//  MIConvolutionLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MetalTensorLayer.h"
#import "MIDataSource.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIConvolutionLayer : MetalTensorLayer <MetalTensorWeights>

@property (nonatomic, assign) MPSImageEdgeMode edgeMode;
@property (nonatomic, assign) MTPaddingMode padding;
@property (nonatomic, assign) KernelShape kernel;
@property (nonatomic, assign) NeuronType neuron;
@property (nonatomic, assign) MTLInt2 offset;
@property (nonatomic, strong) MICNNKernelDataSource *dataSource;
@property (nonatomic, assign, getter=isDepthWise) BOOL depthWise;

MIConvolutionLayer *MakeConvolutionLayer(NSString *module_name,
                                         KernelShape *k,
                                         NeuronType *n,
                                         MTPaddingMode padding,
                                         DataShape *input);

@end

NS_ASSUME_NONNULL_END
