//
//  MIInvertedResidualModule.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/21.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"
#import "MIConvolutionLayer.h"
#import "MIDataSource.h"

NS_ASSUME_NONNULL_BEGIN

/**
 *  NOTE: ONLY support 'tensorflow-same' padding mode.
 *
 **/

@interface MIInvertedResidualModule : MetalTensorLayer <MetalTensorWeights>

@property (nonatomic, strong) MICNNKernelDataSource *expandDataSource;
@property (nonatomic, strong) MICNNKernelDataSource *depthWiseDataSource;
@property (nonatomic, strong) MICNNKernelDataSource *projectDataSource;

@property (nonatomic, readonly) KernelShape *kernels;
@property (nonatomic, readonly) NeuronType *neurons;

- (MIConvolutionLayer *)expandComponent;
- (MIConvolutionLayer *)depthWiseComponent;
- (MIConvolutionLayer *)projectComponent;

MIInvertedResidualModule *MakeInvertedResidualModule(NSString *expand,
                                                     NSString *depthwise,
                                                     NSString *project,
                                                     KernelShape * _Nonnull kernels,
                                                     NeuronType * _Nonnull neurons,
                                                     DataShape *_Nonnull inputShape);

@end

NS_ASSUME_NONNULL_END
