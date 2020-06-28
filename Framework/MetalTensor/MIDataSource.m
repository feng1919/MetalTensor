//
//  MIDataSource.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/22.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIDataSource.h"
#include <stdio.h>
#import <QuartzCore/QuartzCore.h>
#include "numpy.h"

@implementation MIDataSource

- (instancetype)init {
    if (self = [super init]) {
        _range.location = NSNotFound;
        _range.length = 0;
        _purgeEnable = YES;
    }
    return self;
}

- (instancetype)initWithContentOfFile:(NSString *)filePath {
    if (self = [super init]) {
        _filePath = filePath;
        _range.location = NSNotFound;
        _range.length = 0;
        _purgeEnable = YES;
    }
    return self;
}

- (NSData *)data {
    return _data;
}

@end

@implementation MICNNKernelDataSource

- (instancetype)initWithData:(NSData *)weights
                      kernel:(KernelShape *)kernel
                      neuron:(NeuronType *)neuron
                   depthWise:(BOOL)depthWise
{
    if (self = [super init]) {
        _purgeEnable = NO;
        _kernel = kernel[0];
        _neuron = neuron[0];
        _depthWise = depthWise;
        _data = weights;
        _rotateSpatial180 = NO;
        
    }
    return self;
}

- (instancetype)initWithContentOfFile:(NSString *)filePath
                               kernel:(KernelShape *)kernel
                               neuron:(NeuronType *)neuron {
    return [self initWithContentOfFile:filePath kernel:kernel
                                neuron:neuron depthWise:NO];
}

- (instancetype)initWithContentOfFile:(NSString *)filePath
                               kernel:(KernelShape *)kernel
                               neuron:(NeuronType *)neuron
                            depthWise:(BOOL)depthWise
{
    if (self = [super initWithContentOfFile:filePath]) {
        _kernel = kernel[0];
        _neuron = neuron[0];
        _depthWise = depthWise;
        _rotateSpatial180 = NO;
        _transposeIO = NO;
        _removeBias = NO;
    }
    return self;
}

- (NSString *)description {
    NSString *fileName = _filePath.lastPathComponent;
    NSString *kernel = NSStringFromKernelShape(&_kernel);
    NSString *neuron = NSStringFromNeuronType(&_neuron);
    NSString *depthWise = _depthWise?@"dw":@"pw";
    if (_range.location == NSNotFound) {
        return [NSString stringWithFormat:@"<%@, %@, %@, %@>", fileName, depthWise, kernel, neuron];
    }
    
    return [NSString stringWithFormat:@"<%@(%ld, %ld), %@, %@, %@>", fileName, _range.location, _range.length, depthWise, kernel, neuron];
}

- (MPSDataType)dataType {
    return MPSDataTypeFloat32;
}

- (void *)weights {
    return (void *)[_data bytes];
}

- (float32_t *)biasTerms {
    
    if (_removeBias) {
        return NULL;
    }
    
    int size;
    if (_depthWise) {
        size = _kernel.row*_kernel.column*_kernel.depth;
    }
    else {
        size = _kernel.row*_kernel.column*_kernel.depth*_kernel.filters;
    }
    
    NSAssert([_data length] >= size*sizeof(float32_t),
             @"Invalid data layout, data length: %ld, expected size = %ld",
             [_data length], size*sizeof(float32_t));
    
    if ([_data length] == size*sizeof(float32_t)) {
        return NULL;
    }
    else {
        float32_t *p = (float32_t *)[_data bytes];
        return p+size;
    }
}

- (BOOL)load {
    if (_data == nil) {
        NSError *err = nil;
        if (_range.location != NSNotFound) {
            
            NSAssert(_range.length > 0, @"Invalid weights data range length. %@", [self description]);
            
            FILE *fp;
            fp = fopen(_filePath.UTF8String, "r");
            if (fp == NULL) {
                NSAssert(NO, @"Failed to load file: %@", _filePath);
                return NO;
            }
            fseek(fp, _range.location, SEEK_SET);
            Byte *buffer = malloc(_range.length * sizeof(Byte));
            fread(buffer, sizeof(Byte), _range.length, fp);
            if (_rotateSpatial180) {
                rotate_spatial_180((float32_t *)buffer, &_kernel);
            }
            if (_transposeIO) {
                transpose_input_output_channels((float32_t*)buffer, &_kernel);
            }
            _data = [NSData dataWithBytesNoCopy:buffer length:_range.length freeWhenDone:YES];
            fclose(fp);
        }
        else {
            _data = [NSData dataWithContentsOfFile:_filePath options:NSDataReadingMappedIfSafe error:&err];
        }
        NSAssert(err==nil, @"Load layer weights failed. %@\nreason: %@", _filePath, err);
    }
    return _data != nil && _data.length > 0;
}

- (void)purge {
    if (_purgeEnable) {
        NSLog(@"MPS purge weight data. [%@][%ld Bytes]", [self label], [_data length]);
        _data = nil;
    }
}

- (NSString *)label {
    return _label?:[NSString stringWithFormat:@"weight:%ld", [_data length]];
}

- (MPSCNNConvolutionDescriptor *)descriptor {
    if (_descriptor == nil) {
        NSAssert(_kernel.depth>0, @"Invliad depth of Kernels.");
        if (_depthWise) {
            _descriptor = [MPSCNNDepthWiseConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:_kernel.column
                                                                                          kernelHeight:_kernel.row
                                                                                  inputFeatureChannels:_kernel.depth
                                                                                 outputFeatureChannels:_kernel.depth];
        }
        else {
            int depth = _transposeIO?_kernel.filters:_kernel.depth;
            int filters = _transposeIO?_kernel.depth:_kernel.filters;
            _descriptor = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:_kernel.column
                                                                                 kernelHeight:_kernel.row
                                                                         inputFeatureChannels:depth
                                                                        outputFeatureChannels:filters];
        }
    
        [_descriptor setFusedNeuronDescriptor:[MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:_neuron.neuron
                                                                                              a:_neuron.a
                                                                                              b:_neuron.b]];
        [_descriptor setStrideInPixelsX:_kernel.stride];
        [_descriptor setStrideInPixelsY:_kernel.stride];
        
        if (_bn) {
            [_descriptor setBatchNormalizationParametersForInferenceWithMean:[_bn mean]
                                                                   variance:[_bn variance]
                                                                      gamma:[_bn gamma]
                                                                       beta:[_bn beta]
                                                                    epsilon:[_bn epsilon]];
        }
    }
    return _descriptor;
}

- (id)copyWithZone:(NSZone *)zone {
    MICNNKernelDataSource *item = [MICNNKernelDataSource allocWithZone:zone];
    item.filePath = _filePath;
    item.label = _label;
    item.range = _range;
    item->_data = _data;
    item.kernel = _kernel;
    item.neuron = _neuron;
    item.depthWise = _depthWise;
    item.bn = _bn;
    item.rotateSpatial180 = _rotateSpatial180;
    item.transposeIO = _transposeIO;
    return item;
}

- (id)copy {
    MICNNKernelDataSource *item = [[MICNNKernelDataSource alloc] init];
    item.filePath = _filePath;
    item.label = _label;
    item.range = _range;
    item->_data = _data;
    item.kernel = _kernel;
    item.neuron = _neuron;
    item.depthWise = _depthWise;
    item.bn = _bn;
    item.rotateSpatial180 = _rotateSpatial180;
    item.transposeIO = _transposeIO;
    return item;
}

@end

@implementation MIBNParametersDataSource

-(NSUInteger) numberOfFeatureChannels {
    return [_data length] >> 4;
}

-(float * __nullable) gamma {
    return (float *)[_data bytes];
}

-(float * __nullable) beta {
    return (float *)((Byte *)[_data bytes] + [self numberOfFeatureChannels] * sizeof(float32_t));
}

-(float * __nullable) mean {
    return (float *)((Byte *)[_data bytes] + [self numberOfFeatureChannels] * 2 * sizeof(float32_t));
}

-(float * __nullable) variance {
    return (float *)((Byte *)[_data bytes] + [self numberOfFeatureChannels] * 3 * sizeof(float32_t));
}

-(BOOL) load {
    if (_data == nil) {
        NSError *err = nil;
        _data = [NSData dataWithContentsOfFile:_filePath options:NSDataReadingMappedIfSafe error:&err];
        NSAssert(err==nil, @"Load layer weights failed. %@\nreason: %@", _filePath, err);
    }
    return _data != nil && _data.length > 0;
}

-(void) purge {
    NSLog(@"MPS purge weight data. [%@][%ld Bytes]", [self label], [_data length]);
    _data = nil;
}

-(NSString* __nullable) label {
    return _label?:[NSString stringWithFormat:@"weight:%ld", [_data length]];
}

- (float)epsilon {
    return 0.00001f;
}

- (id)copyWithZone:(NSZone *)zone {
    MIBNParametersDataSource *item = [MIBNParametersDataSource allocWithZone:zone];
    item->_filePath = _filePath;
    item->_data = _data;
    return item;
}

- (id)copy {
    MIBNParametersDataSource *item = [[MIBNParametersDataSource alloc] init];
    item->_filePath = _filePath;
    item->_data = _data;
    return item;
}

@end

void transpose_input_output_channels(float32_t *buffer, KernelShape *kernel) {
    const int N = kernel->filters;
    const int CH = kernel->depth;
    const int H = kernel->row;
    const int W = kernel->column;
    
    float32_t *temp = malloc(N*CH*H*W*sizeof(float32_t));
    
    for (int ch = 0; ch < CH; ch++) {
        int off0 = ch * H * W * N;
        for (int row = 0; row < H; row ++) {
            int off1 = row * W * N;
            for (int col = 0; col < W; col++) {
                int off2 = col * N;
                for (int n = 0; n < N; n++) {
                    temp[off0+off1+off2+n] = buffer[n*H*W*CH+row*W*CH+col*CH+ch];
                }
            }
        }
    }
    
    npmemcpy(buffer, temp, N*CH*H*W*sizeof(float32_t));
    
    free(temp);
}

void rotate_spatial_180(float32_t *buffer, KernelShape *kernel) {
    
    const int N = kernel->filters;
    const int CH = kernel->depth;
    const int H = kernel->row;
    const int W = kernel->column;
    
    const int CH_SIZE = CH * sizeof(float32_t);
    
    float32_t *temp = malloc(CH*H*W*sizeof(float32_t));
    for (int i = 0; i < N; i++) {
        int unit = i * H * W * CH;
        for (int row = 0; row < H; row++) {
            for (int col = 0; col < W; col ++) {
                npmemcpy(temp+((H-row)*W-1-col)*CH, buffer+unit+(row*W+col)*CH, CH_SIZE);
            }
        }
        
//        if (i == 0) {
//            printf("\nbefore rotate 180:");
//            for (int row = 0; row < H; row++) {
//                printf("\n[");
//                for (int col = 0; col < W; col++) {
//                    printf("[");
//                    for (int c = 0; c < CH; c++) {
//                        printf("%f", buffer[(row * W + col) * CH + c]);
//                        if (c < CH-1) {
//                            printf(", ");
//                        }
//                    }
//                    printf("]");
//                    if (col < W-1) {
//                        printf("\n");
//                    }
//                }
//                printf("]");
//            }
//        }
        npmemcpy(buffer+unit, temp, CH_SIZE*H*W);
//        if (i == 0) {
//            printf("\nafter rotate 180: ");
//            for (int row = 0; row < H; row++) {
//                printf("\n[");
//                for (int col = 0; col < W; col++) {
//                    printf("[");
//                    for (int c = 0; c < CH; c++) {
//                        printf("%f", buffer[(row * W + col) * CH + c]);
//                        if (c < CH-1) {
//                            printf(", ");
//                        }
//                    }
//                    printf("]");
//                    if (col < W-1) {
//                        printf("\n");
//                    }
//                }
//                printf("]");
//            }
//        }
    }
    free(temp);
}

MICNNKernelDataSource *MakeDataSource(NSString *module_name, KernelShape *k, NeuronType *n)
{
    NSString *w_file = [[NSBundle mainBundle] pathForResource:module_name ofType:@"bin"];
    assert([[NSFileManager defaultManager] fileExistsAtPath:w_file]);
    BOOL isDepthWise = (([[module_name lowercaseString] rangeOfString:@"dw"].location != NSNotFound) ||
                        ([[module_name lowercaseString] rangeOfString:@"depthwise"].location != NSNotFound));
    return MakeDataSource2(w_file, k, n, isDepthWise, NULL);
}

MICNNKernelDataSource *MakeDataSource1(NSString *module_name, KernelShape *k, NeuronType *n, BOOL isDepthWise)
{
    NSString *w_file = [[NSBundle mainBundle] pathForResource:module_name ofType:@"bin"];
    assert([[NSFileManager defaultManager] fileExistsAtPath:w_file]);
    return MakeDataSource2(w_file, k, n, isDepthWise, NULL);
}

MICNNKernelDataSource *MakeDataSource2(NSString *w_file, KernelShape *k, NeuronType *n, BOOL isDepthWise, NSRange *range)
{
//    printf("\n Load Weights: %s [kernel:%dx%dx%d][neuron:%d,%f,%f][dw:%d]", [[w_file lastPathComponent] UTF8String],
//           k->row, k->column, k->depth, n->neuron, n->a, n->b, isDepthWise);
    
    MICNNKernelDataSource *dataSource = [[MICNNKernelDataSource alloc] initWithContentOfFile:w_file
                                                                                      kernel:k
                                                                                      neuron:n
                                                                                   depthWise:isDepthWise];
    dataSource.label = [[w_file lastPathComponent] stringByDeletingPathExtension];
    if (range) {
        dataSource.range = *range;
    }
    return dataSource;
}
