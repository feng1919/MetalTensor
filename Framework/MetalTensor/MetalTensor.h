//
//  MetalTensor.h
//  MetalTensor
//
//  Created by Feng Stone on 2019/9/30.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalImage/MetalImage.h>

//! Project version number for MetalTensor.
FOUNDATION_EXPORT double MetalTensorVersionNumber;

//! Project version string for MetalTensor.
FOUNDATION_EXPORT const unsigned char MetalTensorVersionString[];


#import "MTTensor.h"
#import "MTTensorCache.h"
#import "MTImageTensor.h"
#import "MetalTensorProtocols.h"

#import "numpy.h"
#import "metal_tensor_log.h"
#import "metal_tensor_structures.h"

#import "FPSCounter.h"
#import "MetalLayerSwitch.h"
#import "MetalNeuralNetwork.h"
#import "MetalTensorInputLayer.h"
#import "MetalTensorLayer.h"
#import "MetalTensorLayerDescriptor.h"
#import "MetalTensorNeuronLayer.h"
#import "MetalTensorNode.h"
#import "MetalTensorOutputLayer.h"
#import "MIArithmeticLayer.h"
#import "MIBatchNormalizationLayer.h"
#import "MIConcatenateLayer.h"
#import "MIConvolutionLayer.h"
#import "MIDataSource.h"
#import "MIDropoutLayer.h"
#import "MIFullyConnectedLayer.h"
#import "MIInceptionV3Module.h"
#import "MIInvertedResidualModule.h"
#import "MIL2NormalizationLayer.h"
#import "MIPoolingAverageLayer.h"
#import "MIPoolingMaxLayer.h"
#import "MIReshapeLayer.h"
#import "MIResidualModule.h"
#import "MISeparableConvolutionLayer.h"
#import "MISoftMaxLayer.h"
#import "MITransposeConvolutionLayer.h"
#import "MIMatrixMultiplyLayer.h"
#import "MIGramMatrixLayer.h"
#import "MIReduceUnaryLayer.h"
#import "MTMeanSquaredErrorLayer.h"
#import "MetalTensorSpacialReduce.h"
#import "MTChannelReduce.h"
#import "MetalTensorSlice.h"
#import "MTTotalVariationLayer.h"
#import "MTGramMatrixLayer.h"


#import "MPSImage+Extension.h"
#import "NSString+Extension.h"

// SSD decoding
#import "ssd_decoder.h"
#import "SSDConfig.h"
#import "SSDObject.h"
