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
#import "MetalTensorInput.h"
#import "MIMPSImage.h"
#import "MIPoolingAverageLayer.h"
#import "MIPoolingMaxLayer.h"
#import "MIReshapeLayer.h"
#import "MIResidualModule.h"
#import "MISeparableConvolutionLayer.h"
#import "MISoftMaxLayer.h"
#import "MITemporaryImage.h"
#import "MITemporaryImageCache.h"
#import "MITransposeConvolutionLayer.h"
#import "MPSImage+Extension.h"
#import "NSString+Extension.h"
