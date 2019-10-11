//
//  metal_tensor_structures.c
//  MetalImage
//
//  Created by Feng Stone on 2019/6/26.
//  Copyright © 2019 fengshi. All rights reserved.
//

#include "metal_tensor_structures.h"
#include <stdbool.h>
#include <assert.h>
#include <math.h>

bool DataShapeValid(DataShape *s) {
    return (s->row>0 &&
            s->column>0 &&
            s->depth>0);
}

bool DataShapesTheSame(DataShape *s1, DataShape *s2) {
    return (s1->depth==s2->depth &&
            s1->row==s2->row &&
            s1->column==s2->column);
}

bool KernelShapeValid(KernelShape *s) {
    return (s->row>0 &&
            s->column>0 &&
            s->depth>0 &&
            s->filters>0 &&
            s->stride>0);
}

int ProductOfDataShape(DataShape *s) {
    return s->column * s->depth * s->row;
}

int ProductOfDataShapeDepth4Divisible(DataShape *s) {
    return s->column*s->row*((s->depth+3)>>2)<<2;
}

DataShape Reshape(DataShape *shape, int row, int column, int depth) {
    assert(ProductOfDataShape(shape) > 0);
    DataShape s;
    s.column = shape->column;
    s.depth = shape->depth;
    s.row = shape->row;
    int result = Reshape1(&s, &row, &column, &depth);
    return result==0?s:*shape;
}

int Reshape1(DataShape *shape, int *row, int *column, int *depth) {
    int unknown = -1;
    int product = 1;
    if (*row <= 0) {
        if (unknown == -1) {
            unknown = 0;
        }
        else {
            goto InvalidArgument;
        }
    }
    else {
        product = product**row;
    }
    
    if (*column <= 0) {
        if (unknown == -1) {
            unknown = 1;
        }
        else {
            goto InvalidArgument;
        }
    }
    else {
        product = product**column;
    }
    
    if (*depth <= 0) {
        if (unknown == -1) {
            unknown = 2;
        }
        else {
            goto InvalidArgument;
        }
    }
    else {
        product = product**depth;
    }
    
    if (unknown == -1) {
        if (ProductOfDataShape(shape) == product) {
            shape->row = *row;
            shape->column = *column;
            shape->depth = *depth;
            return 0;
        }
        else {
            goto InvalidArgument;
        }
    }
    
    int original = ProductOfDataShape(shape);
    if (original%product != 0) {
        goto InvalidArgument;
    }
    if (unknown == 0) {
        *row = original/product;
    }
    else if (unknown == 1) {
        *column = original/product;
    }
    else if (unknown == 2) {
        *depth = original/product;
    }
    shape->row = *row;
    shape->column = *column;
    shape->depth = *depth;
    return 0;
    
InvalidArgument:
    printf("\nFailed to reshape, only support one missing dimension\
           and the divid remainder of product must be zero.");
    return -1;
}

DataShape DataShapeTranspose(DataShape *shape, int row_i, int column_i, int depth_i) {
    DataShape data_shape;
    if (row_i == 1) {
        data_shape.row = shape->column;
    }
    else if (row_i == 2) {
        data_shape.row = shape->depth;
    }
    else {
        data_shape.row = shape->row;
    }
    
    if (column_i == 0) {
        data_shape.column = shape->row;
    }
    else if (column_i == 2) {
        data_shape.column = shape->depth;
    }
    else {
        data_shape.column = shape->column;
    }
    
    if (depth_i == 0) {
        data_shape.depth = shape->row;
    }
    else if (depth_i == 1) {
        data_shape.depth = shape->column;
    }
    else {
        data_shape.depth = shape->depth;
    }
    
    return data_shape;
}

bool KernelShapesTheSame(KernelShape *s1, KernelShape *s2) {
    return (s1->row==s2->row &&
            s1->column==s2->column &&
            s1->depth==s2->depth &&
            s1->filters==s2->filters &&
            s1->stride==s2->stride);
}

DataShape DataShapeMake(int row, int column, int depth) {
    DataShape s;
    s.row = row;
    s.column = column;
    s.depth = depth;
    return s;
}

DataShape ConcatenateShapes(DataShape *shapes, int size, int *offsets, bool make_divisible_4) {
    assert(shapes != NULL && size > 0);
    int maxRow = 0;
    int maxColumn = 0;
    int totalDepth = 0;
    
    for (int i = 0; i < size; i++) {
        if (offsets != NULL) {
            offsets[i] = totalDepth;
        }
        
        maxRow = fmaxf(maxRow, shapes[i].row);
        maxColumn = fmaxf(maxColumn, shapes[i].column);
        int depth = shapes[i].depth;
        totalDepth += (make_divisible_4?(((depth+3)>>2)<<2):depth);
    }
    return DataShapeMake(maxRow, maxColumn, totalDepth);
}

DataShape ConcatenateShapes1(DataShape **shapes, int size, int *offsets, bool make_divisible_4) {
    assert(shapes != NULL && size > 0);
    int maxRow = 0;
    int maxColumn = 0;
    int totalDepth = 0;
    
    for (int i = 0; i < size; i++) {
        if (offsets != NULL) {
            offsets[i] = totalDepth;
        }
        
        maxRow = fmaxf(maxRow, shapes[i][0].row);
        maxColumn = fmaxf(maxColumn, shapes[i][0].column);
        int depth = shapes[i][0].depth;
        totalDepth += (make_divisible_4?(((depth+3)>>2)<<2):depth);
    }
    return DataShapeMake(maxRow, maxColumn, totalDepth);
}

KernelShape KernelShapeMake(int row, int column, int depth, int kernel, int stride) {
    KernelShape k;
    k.row = row;
    k.column = column;
    k.depth = depth;
    k.filters = kernel;
    k.stride = stride;
    return k;
}

int make_divisible_8(int v) {
    return make_divisible(v, 8, 8);
}

int make_divisible(int v, int divisor, int min_value) {
    int new_v = fmaxf(min_value, roundf((float)v/(float)divisor) * divisor);
    // Make sure that round down does not go down by more than 10%.
    if (new_v < 0.9f*v) {
        new_v += divisor;
    }
    return new_v;
}