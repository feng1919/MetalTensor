//
//  MPSImage+Extension.m
//  MPSToolsKit
//
//  Created by Feng Stone on 2019/5/17.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MPSImage+Extension.h"
#import <Accelerate/Accelerate.h>
#include "numpy.h"

@implementation MPSImage (Extension)

bool ConvertFloat32To16(float32_t *src, float16_t *dst, unsigned int size) {
    vImage_Buffer f16;
    f16.data = dst;
    f16.height = 1;
    f16.width = size;
    f16.rowBytes = size * sizeof(float16_t);
    
    vImage_Buffer f32;
    f32.data = src;
    f32.height = 1;
    f32.width = size;
    f32.rowBytes = size * sizeof(float32_t);
    
    return vImageConvert_PlanarFtoPlanar16F(&f32, &f16, 0) == kvImageNoError;
}

bool ConvertFloat16To32(float16_t *src, float32_t *dst, unsigned int size) {
    vImage_Buffer f16;
    f16.data = src;
    f16.height = 1;
    f16.width = size;
    f16.rowBytes = size * sizeof(float16_t);
    
    vImage_Buffer f32;
    f32.data = dst;
    f32.height = 1;
    f32.width = size;
    f32.rowBytes = size * sizeof(float32_t);
    
    return vImageConvert_Planar16FtoPlanarF(&f16, &f32, 0) == kvImageNoError;
}

void ConvertKernelFirstToLast(float32_t *src, float32_t *dst, int col, int row, int depth, int kernel) {
    int step = col*row*depth;
    for (int r = 0; r < kernel; r++) {
        for (int c = 0; c < step; c++) {
            dst[r*step+c] = src[c*kernel+r];
        }
    }
}

void ConvertToTensorFlowLayout(float *dst, const float *src, const DataShape *shape)
{
    if (shape->depth <= 4) {
        npmemcpy(dst, src, ProductDepth4Divisible(shape)*sizeof(float));
        return;
    }
    
    int unit = shape->column*shape->row*4;
    for (int i = 0; i < shape->row; i++) {
        int offset_y = i*shape->column*shape->depth;
        for (int j = 0; j < shape->column; j++) {
            int offset = j*shape->depth+offset_y;
            for (int k = 0; k < shape->depth; k++) {
                dst[offset+k] = src[unit*(k>>2)+(i*shape->column+j)*4+(k&0x03)];
            }
        }
    }
}


void ConvertToTensorFlowLayout1(float *src, const DataShape *shape) {
    
    if (shape->depth <= 4) {
        return;
    }
    
    unsigned long long buffer_size = ProductDepth4Divisible(shape);
    float *temp = malloc(buffer_size * sizeof(float));
    ConvertToTensorFlowLayout(temp, src, shape);
    npmemcpy(src, temp, buffer_size*sizeof(float));
    free(temp);
}


void ConvertF16ToTensorFlowLayout(float16_t *dst, const float16_t *src, const DataShape *shape)
{
    double interval = CFAbsoluteTimeGetCurrent();
    if (shape->depth <= 4) {
        npmemcpy(dst, src, ProductDepth4Divisible(shape)*sizeof(float16_t));
        return;
    }
    
    int unit = shape->column*shape->row*4;
    for (int i = 0; i < shape->row; i++) {
        int offset_y = i*shape->column*shape->depth;
        for (int j = 0; j < shape->column; j++) {
            int offset = j*shape->depth+offset_y;
            for (int k = 0; k < shape->depth; k++) {
                dst[offset+k] = src[unit*(k>>2)+(i*shape->column+j)*4+(k&0x03)];
            }
        }
    }
    printf("\nconvert to HWC: %f", (CFAbsoluteTimeGetCurrent()-interval)*1000.0f);
}

void ConvertF16ToTensorFlowLayout1(float16_t *src, const DataShape *shape)
{
    if (shape->depth <= 4) {
        return;
    }
    
    unsigned long long buffer_size = ProductDepth4Divisible(shape);
    float16_t *temp = malloc(buffer_size * sizeof(float16_t));
    ConvertF16ToTensorFlowLayout(temp, src, shape);
    npmemcpy(src, temp, buffer_size*sizeof(float16_t));
    free(temp);
}

void ConvertF16ToCHWLayout(float16_t *dst, const float16_t *src, const DataShape *shape) {
    const int rows = shape->row;
    const int columns = shape->column;
    const int depth = shape->depth;
    const int stride = rows*columns;
    
    for (int dep = 0; dep < depth; dep++) {
        int offset = (dep>>2) * stride * 4;
        int offset_dep = dep % 4;
        int texture_dep = MIN(4, depth-((dep>>2)<<2));
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < columns; col++) {
                dst[dep*stride+row*columns+col] = src[offset+(row*columns+col)*texture_dep+offset_dep];
            }
        }
    }
}

void ConvertF16ToCHWLayout1(float16_t *src, const DataShape *shape) {
    unsigned long long buffer_size = ProductDepth4Divisible(shape);
    float16_t *temp = malloc(buffer_size * sizeof(float16_t));
    ConvertF16ToCHWLayout(temp, src, shape);
    npmemcpy(src, temp, buffer_size*sizeof(float16_t));
    free(temp);
}

void ConvertF16HWCToCHWLayout(float16_t *dst, const float16_t *src, const DataShape *shape) {
    double interval = CFAbsoluteTimeGetCurrent();
    const int rows = shape->row;
    const int columns = shape->column;
    const int depth = shape->depth;
    const int stride = rows*columns;
    
    for (int dep = 0; dep < depth; dep++) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < columns; col++) {
                dst[dep*stride+row*columns+col] = src[(row*columns+col)*depth+dep];
            }
        }
    }
    printf("\nconvert HWC to CHW time cost: %f", (CFAbsoluteTimeGetCurrent()-interval)*1000.0f);
}

void ConvertF16HWCToCHWLayout1(float16_t *src, const DataShape *shape) {
    unsigned long long buffer_size = ProductDepth4Divisible(shape);
    float16_t *temp = malloc(buffer_size * sizeof(float16_t));
    ConvertF16HWCToCHWLayout(temp, src, shape);
    npmemcpy(src, temp, buffer_size*sizeof(float16_t));
    free(temp);
}

NeuronType NeuronTypeMake(MPSCNNNeuronType n, float a, float b) {
    NeuronType s;
    s.neuron = n;
    s.a = a;
    s.b = b;
    s.c = 0.0f;
    return s;
}

NeuronType NeuronTypeMake1(MPSCNNNeuronType n, float a, float b, float c) {
    NeuronType s;
    s.neuron = n;
    s.a = a;
    s.b = b;
    s.c = c;
    return s;
}

NeuronType NeuronTypeNone(void) {
    return NeuronTypeMake(MPSCNNNeuronTypeNone, 0.0f, 0.0f);
}

NeuronType Relu(void) {
    return NeuronTypeMake(MPSCNNNeuronTypeReLU, 0.0f, 0.0f);
}

NeuronType Relu6(void) {
    return NeuronTypeMake(MPSCNNNeuronTypeReLUN, 0.0f, 6.0f);
}

MPSCNNNeuronType NeuronTypeFromString(NSString *neuron)
{
    if ([neuron isEqualToString:@"relu"]) {
        return MPSCNNNeuronTypeReLU;
    }
    if ([neuron isEqualToString:@"linear"]) {
        return MPSCNNNeuronTypeLinear;
    }
    if ([neuron isEqualToString:@"relun"]) {
        return MPSCNNNeuronTypeReLUN;
    }
    if ([neuron isEqualToString:@"sigmod"]) {
        return MPSCNNNeuronTypeSigmoid;
    }
    if ([neuron isEqualToString:@"none"]) {
        return MPSCNNNeuronTypeNone;
    }
    if ([neuron isEqualToString:@"power"]) {
        return MPSCNNNeuronTypePower;
    }
    assert(0);
    return MPSCNNNeuronTypeNone;
}

NSString *StringWithNeuronType(MPSCNNNeuronType neuron)
{
    if (neuron == MPSCNNNeuronTypeReLU) {
        return @"relu";
    }
    if (neuron == MPSCNNNeuronTypeLinear) {
        return @"linear";
    }
    if (neuron == MPSCNNNeuronTypeReLUN) {
        return @"relun";
    }
    if (neuron == MPSCNNNeuronTypeSigmoid) {
        return @"sigmod";
    }
    if (neuron == MPSCNNNeuronTypeNone) {
        return @"none";
    }
    assert(0);
    return @"none";
}

MTPaddingMode PaddingModeFromString(NSString *padding) {
    if ([padding isEqualToString:@"same"]) {
        return MTPaddingMode_tfsame;
    }
    if ([padding isEqualToString:@"valid"]) {
        return MTPaddingMode_valid;
    }
    if ([padding isEqualToString:@"full"]) {
        return MTPaddingMode_full;
    }
    assert(0);
    return MTPaddingMode_tfsame;
}

NSString *NSStringFromDataShape(DataShape *dataShape) {
    return [NSString stringWithFormat:@"(%d, %d, %d)", dataShape->row, dataShape->column, dataShape->depth];
}

NSString *NSStringFromKernelShape(KernelShape *kernel) {
    return [NSString stringWithFormat:@"(%d, %d, %d, %d, %d)", kernel->row, kernel->column, kernel->depth, kernel->filters, kernel->stride];
}

NSString *NSStringFromNeuronType(NeuronType *neuron) {
    return [NSString stringWithFormat:@"(%@, %f, %f, %f)", StringWithNeuronType(neuron->neuron), neuron->a, neuron->b, neuron->c];
}

NSString *NSStringFromRegion(MTLRegion region) {
    return [NSString stringWithFormat:@"(%d, %d, %d, %d, %d, %d)", (int)region.origin.x, (int)region.origin.y,
            (int)region.origin.z, (int)region.size.width, (int)region.size.height, (int)region.size.depth];
}

/*
 We receive the predicted output as an MPSImage. We need to convert this
 to an array of Floats that we can use from Swift.
 
 Because Metal is a graphics API, MPSImage stores the data in MTLTexture
 objects. Each pixel from the texture stores 4 channels: R contains the
 first channel, G is the second channel, B is the third, A is the fourth.
 
 In addition, these individual R,G,B,A pixel components can be stored as
 float16, in which case we also have to convert the data type.
 
 ---WARNING---
 
 If there are > 4 channels in the MPSImage, then the channels are organized
 in the output as follows:
 
 [ 1,2,3,4,1,2,3,4,...,1,2,3,4,5,6,7,8,5,6,7,8,...,5,6,7,8 ]
 
 and not as you'd expect:
 
 [ 1,2,3,4,5,6,7,8,...,1,2,3,4,5,6,7,8,...,1,2,3,4,5,6,7,8 ]
 
 So first are channels 1 - 4 for the entire image, followed by channels
 5 - 8 for the entire image. That happens because we copy the data out of
 the texture by slice, and we can't interleave slices.
 
 If the number of channels is not a multiple of 4, then the output will
 have padding bytes in it:
 
 [ 1,2,3,4,1,2,3,4,...,1,2,3,4,5,6,-,-,5,6,-,-,...,5,6,-,- ]
 
 The only case where you get the kind of array you'd actually expect is
 when the number of channels is 1, 2, or 4 (i.e. there is only one slice):
 
 [ 1,1,1,...,1] or [ 1,2,1,2,1,2,...,1,2 ] or [ 1,2,3,4,...,1,2,3,4 ]
 */

- (unsigned int)sizeOfComponent {
    switch (self.pixelFormat) {
        case MTLPixelFormatRGBA8Sint:
        case MTLPixelFormatBGRA8Unorm:
            return sizeof(int8_t);
            break;
            
        case MTLPixelFormatR16Float:
        case MTLPixelFormatRG16Float:
        case MTLPixelFormatRGBA16Float:
            return sizeof(float16_t);
            break;
            
        case MTLPixelFormatR32Float:
        case MTLPixelFormatRG32Float:
        case MTLPixelFormatRGBA32Float:
            return sizeof(float32_t);
            break;
            
        default:
            NSAssert(NO, @"Unsupported Pixel Format...");
            return 1;
            break;
    }
}

- (void)toFloat16Array:(float16_t *)buffer {
    int numOfSlices = ((int)self.featureChannels + 3)/4;
    
    int numOfComponents = [self numberOfComponents];
    NSUInteger bytesPerSlice = numOfComponents * self.width * self.height;// * [self sizeOfComponent];
    
    for (int i = 0; i < numOfSlices; i++) {
        [self toFloat16Array:buffer+bytesPerSlice*i slice:i];
    }
}

- (void)toFloat16Array:(float16_t *)buffer slice:(int)slice {
    
    int numOfComponents = [self numberOfComponents];
    int sizeOfComponent = [self sizeOfComponent];
    
    MTLRegion region = MTLRegionMake3D(0, 0, 0, self.width, self.height, 1);
    [self.texture getBytes:(void *)buffer
               bytesPerRow:self.width*numOfComponents*sizeOfComponent
             bytesPerImage:0
                fromRegion:region
               mipmapLevel:0
                     slice:slice];
}

- (void)toBuffer:(Byte *)buffer {
    int numOfSlices = ((int)self.featureChannels + 3)/4;
    
    int numOfComponents = [self numberOfComponents];
    NSUInteger bytesPerSlice = numOfComponents * self.width * self.height * [self sizeOfComponent];
    
    for (int i = 0; i < numOfSlices; i++) {
        [self toBuffer:buffer+bytesPerSlice*i slice:i];
    }
}

- (void)toBuffer:(Byte *)buffer slice:(int)slice {
    
    int numOfComponents = [self numberOfComponents];
    int sizeOfComponent = [self sizeOfComponent];
    
    MTLRegion region = MTLRegionMake3D(0, 0, 0, self.width, self.height, 1);
    [self.texture getBytes:buffer
               bytesPerRow:self.width*numOfComponents*sizeOfComponent
             bytesPerImage:0
                fromRegion:region
               mipmapLevel:0
                     slice:slice];
}

- (int)numberOfComponents {
    return self.featureChannels<3?:4;
}

+ (void)texture:(id<MTLTexture>)texture toFloat16Array:(float16_t *)buffer {
    int width = (int)texture.width;
    int height = (int)texture.height;
    
    int numOfComponents = 4;
    int sizeOfComponent = 2;
    
    MTLRegion region = MTLRegionMake3D(0, 0, 0, width, height, 1);
    [texture getBytes:(void *)buffer
          bytesPerRow:width*numOfComponents*sizeOfComponent
        bytesPerImage:0
           fromRegion:region
          mipmapLevel:0
                slice:0];

}

MPSOffset MPSOffsetMake(NSInteger x, NSInteger y, NSInteger z) {
    MPSOffset offset = {x, y, z};
    return offset;
}

@end
