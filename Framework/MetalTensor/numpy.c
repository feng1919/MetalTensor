//
//  numpy.c
//  MetalImage
//
//  Created by Feng Stone on 2019/6/21.
//  Copyright © 2019 fengshi. All rights reserved.
//

#include "numpy.h"
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <stddef.h>
#include <math.h>

int linspace(float start, float stop, int num, bool end_point, float *dst, float *s)
{
    assert(num > 0);
    assert(dst != NULL);
    if (dst == NULL) {
        return -1;
    }
    if (num == 1) {
        dst[0] = start;
        return 1;
    }
    float step = end_point?((stop-start)/(float)(num-1)):((stop-start)/(float)num);
    if (s != NULL) {
        *s = step;
    }
    for (int i = 0; i < num; i++) {
        dst[i] = i * step + start;
    }
    return num;
}

void meshgrid(float *x, int nx, float *y, int ny, float *dx, float *dy)
{
    assert(x != NULL);
    assert(y != NULL);
    assert(dx != NULL);
    assert(dy != NULL);
    assert(nx > 0);
    assert(ny > 0);
    
    for (int i = 0; i < nx; i ++) {
        for (int j = 0; j < ny; j ++) {
            dx[i*ny+j] = x[i];
            dy[j*nx+i] = y[j];
        }
    }
}

void *npmemcpy(void *dest, const void *src, size_t n)
{
    char *dp = dest;
    const char *sp = src;
    while (n--)
        *dp++ = *sp++;
    return dest;
}

void soft_max(float *x, int size) {
    assert(size > 0);
    float max = x[0];
    for (int i = 1; i < size; i++) {
        max = fmaxf(max, x[i]);
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] -= max;
        x[i] = expf(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

int argmax(float *x, int size) {
    assert(size>0);
    
    if (size == 1) {
        return 0;
    }
    
    int max = x[0];
    int index = 0;
    for (int i = 1; i < size; i++) {
        if (x[i] > max) {
            max = x[i];
            index = i;
        }
    }
    
    return index;
}

