//
//  metal_tensor_structures.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/26.
//  Copyright © 2019 fengshi. All rights reserved.
//

#ifndef metal_tensor_structures_h
#define metal_tensor_structures_h

#include <stdio.h>
#include <stdbool.h>
#include "metal_tensor_log.h"

typedef struct DataShape{
    int row;
    int column;
    int depth;
}DataShape;
DataShape DataShapeMake(int row, int column, int depth);
DataShape ConcatenateShapes(DataShape *shapes, int size, int *offsets, bool make_divisible_4);
DataShape ConcatenateShapes1(DataShape **shapes, int size, int *offsets, bool make_divisible_4);
DataShape Reshape(DataShape *shape, int row, int column, int depth);
DataShape DataShapeTranspose(DataShape *shape, int row_i, int column_i, int depth_i);
int Reshape1(DataShape *shape, int *row, int *column, int *depth);
bool DataShapeValid(DataShape *s);
bool DataShapesTheSame(DataShape *s1, DataShape *s2);
int ProductOfDataShape(DataShape *shape);
int ProductOfDataShapeDepth4Divisible(DataShape *shape);



typedef struct KernelShape{
    int row;
    int column;
    int depth;
    int filters;
    int stride;
}KernelShape;
KernelShape KernelShapeMake(int row, int column, int depth, int filters, int stride);
bool KernelShapeValid(KernelShape *s);
bool KernelShapesTheSame(KernelShape *s1, KernelShape *s2);


typedef enum MTPaddingMode{
    MTPaddingMode_tfsame = 0,
    MTPaddingMode_valid = -1,
    MTPaddingMode_full = 1,
}MTPaddingMode;
int conv_output_length(int input_length, int kernel, int stride, MTPaddingMode padding);
int conv_offset(int kernel, int stride, MTPaddingMode padding);
int trans_conv_output_length(int input_length, int kernel, int stride, MTPaddingMode padding);
int trans_conv_offset(int kernel, int stride, MTPaddingMode padding);

int pooling_offset(int kernel);

int make_divisible_8(int v);
int make_divisible(int v, int divisor, int min_value);

#endif /* metal_tensor_structures_h */
