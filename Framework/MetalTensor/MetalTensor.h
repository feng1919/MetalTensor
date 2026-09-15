//
//  MetalTensor.h
//  MetalTensor
//
//  Created by Feng Stone on 2019/9/30.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#if __has_include(<MetalImage/MetalImage.h>)
#import <MetalImage/MetalImage.h>
#else
#import "MetalImage.h"
#endif

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
#import "MIConcatenateLayer.h"
#import "MIConvolutionLayer.h"
#import "MIDataSource.h"
#import "MIFullyConnectedLayer.h"
#import "MIInvertedResidualModule.h"
#import "MIPoolingAverageLayer.h"
#import "MIPoolingMaxLayer.h"
#import "MIReshapeLayer.h"
#import "MISeparableConvolutionLayer.h"
#import "MISoftMaxLayer.h"
#import "MITransposeConvolutionLayer.h"


#import "MPSImage+Extension.h"
#import "NSString+Extension.h"

// SSD decoding
#import "ssd_decoder.h"
#import "SSDConfig.h"
#import "SSDObject.h"
