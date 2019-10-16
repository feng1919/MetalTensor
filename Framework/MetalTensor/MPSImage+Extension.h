//
//  MPSImage+Extension.h
//  MPSToolsKit
//
//  Created by Feng Stone on 2019/5/17.
//  Copyright © 2019 fengshi. All rights reserved.
//

/*
 * Reference From:
 * https://github.com/hollance/TensorFlow-iOS-Example/blob/master/VoiceMetal/VoiceMetal/
 */

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "metal_tensor_structures.h"

NS_ASSUME_NONNULL_BEGIN

@interface MPSImage (Extension)

typedef struct {
    MPSCNNNeuronType neuron;
    float a;
    float b;
    float c;
}NeuronType;
NeuronType NeuronTypeMake(MPSCNNNeuronType n, float a, float b);
NeuronType NeuronTypeMake1(MPSCNNNeuronType n, float a, float b, float c);

MTPaddingMode PaddingModeFromString(NSString *padding);

MPSCNNNeuronType NeuronTypeFromString(NSString *neuron);
NSString *StringWithNeuronType(MPSCNNNeuronType neuron);
NSString *NSStringFromDataShape(DataShape *dataShape);
NSString *NSStringFromKernelShape(KernelShape *kernel);
NSString *NSStringFromNeuronType(NeuronType *neuron);

bool ConvertFloat32To16(float32_t *src, float16_t *dst, unsigned int size);
bool ConvertFloat16To32(float16_t *src, float32_t *dst, unsigned int size);
void ConvertKernelFirstToLast(float32_t *src, float32_t *dst, int col, int row, int depth, int kernel);

/*
 *  Convert the data layout from Metal to tensorflow.
 */
void ConvertToTensorFlowLayout(float *dst, float *src, DataShape *shape);

- (unsigned int)sizeOfComponent;
- (void)toFloat16Array:(float16_t *)buffer;
- (void)toFloat16Array:(float16_t *)buffer slice:(int)slice;
+ (void)texture:(id<MTLTexture>)texture toFloat16Array:(float16_t *)buffer;

@end

NS_ASSUME_NONNULL_END
