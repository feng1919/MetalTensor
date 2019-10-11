//
//  numpy.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/21.
//  Copyright © 2019 fengshi. All rights reserved.
//

#ifdef __cplusplus
extern "C" {
#endif
    
#ifndef numpy_h
#define numpy_h
    
#include <stdio.h>
#include <stdbool.h>
    
    int linspace(float start, float stop, int num, bool end_point, float *dst, float *step);
    void meshgrid(float *x, int nx, float *y, int ny, float *dx, float *dy);
    void *npmemcpy(void *dest, const void *src, size_t n);
    void soft_max(float *x, int size);
    int argmax(float *x, int size);
    
#endif /* numpy_h */
    
    
#ifdef __cplusplus
} // extern "C"
#endif
