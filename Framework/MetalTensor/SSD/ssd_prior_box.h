//
//  prior_box.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/25.
//  Copyright © 2019 fengshi. All rights reserved.
//

#ifndef prior_box_h
#define prior_box_h

#include <stdio.h>
#include <stdbool.h>

typedef struct {
    float aspect_ratios[8];
    int count;
}pb_ratios;

typedef struct {
    float *cx;
    float *cy;
    int nx;
    int ny;
}pb_meshgrid;

typedef struct {
    float coords[4];
}pb_rect;

typedef struct {
    float width;
    float height;
}pb_size;

typedef struct {
    float x;
    float y;
}pb_point;

typedef struct {
    float dx;
    float dy;
}pb_vector;

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

typedef struct prior_box{
//    pb_size image_size;
//    pb_size feature_map_size;
//    float this_scale;
//    float next_scale;
//    bool normalize_coords; // false
//    bool clip_boxes; // false
//    pb_ratios ratios; // [0.5, 1.0, 2.0]
//    pb_vector steps;
//    pb_vector offsets; // (0.5, 0.5)
    
    int n_ars;
    int n_boxes;
//    pb_size *box_sizes;
    pb_meshgrid meshgrid;
    pb_rect *boxes;
}prior_box;

prior_box *prior_box_create(pb_size *image_size, pb_size *feature_map_size, char coords,
                            float *this_scale, float *next_scale, pb_ratios *ratios,
                            bool two_boxes_for_ar1, bool normalize_coords, bool clip_boxes,
                            pb_vector *steps, pb_vector *offsets);
void prior_box_destroy(prior_box *box);
int number_of_ars(pb_ratios *aspect_ratios, bool two_boxes_for_ar1);
void log_the_boxes(prior_box *box);

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

typedef struct ssd_prior_box{
    pb_size image_size;
    pb_size *feature_map_sizes;
    int n_pyramid_layres;
    float *scales;
    pb_ratios *ratios;
    bool two_boxes_for_ar1;
    bool normalize_coords;
    bool clip_boxes;
    pb_vector *steps;
    pb_vector *offsets;
    
    int n_boxes;
    prior_box **layers;
    
}ssd_prior_box;
ssd_prior_box *ssd_prior_box_create(pb_size *image_size, int n_pyramid_layers, pb_size *feature_map_sizes,
                                    float *scales, pb_ratios *ratios, char coords,
                                    bool two_boxes_for_ar1, bool normalize_coords, bool clip_boxes,
                                    pb_vector *steps, pb_vector *offsets);
void ssd_prior_box_get_all_boxes(pb_rect *dst, ssd_prior_box *buffer);
void ssd_prior_box_destroy(ssd_prior_box *box);

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
// converting coords between minmax and centroids

void ssd_minmax_to_centroids(float *cx, float *cy, float *w, float *h,
                            float xmin, float xmax, float ymin, float ymax);
void ssd_centroids_to_minmax(float *xmin, float *xmax, float *ymin, float *ymax,
                             float cx, float cy, float w, float h);

#endif /* prior_box_h */
