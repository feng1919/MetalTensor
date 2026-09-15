//
//  prior_box.c
//  MetalImage
//
//  Created by Feng Stone on 2019/6/25.
//  Copyright © 2019 fengshi. All rights reserved.
//

#include "ssd_prior_box.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "numpy.h"

void make_pb_sizes(pb_size *sizes, int *n_ars, pb_ratios *aspect_ratios, bool two_boxes_for_ar1,
                   pb_size *image_size, float *this_scale, float *next_scale);
void make_pb_meshgrid(pb_meshgrid *meshgrid, pb_vector *offsets, pb_vector *steps);
void make_prior_boxes(pb_rect *boxes, pb_meshgrid *meshgrid, int n_ars, pb_size *sizes, char coords);
void normalize_box(pb_rect *rect, pb_size *image_size, char coords);
void clip_box(pb_rect *rect, pb_size *image_size, char coords);

prior_box *prior_box_create(pb_size *image_size, pb_size *feature_map_size, char coords,
                            float *this_scale, float *next_scale, pb_ratios *ratios1,
                            bool two_boxes_for_ar1, bool normalize_coords, bool clip_boxes,
                            pb_vector *steps1, pb_vector *offsets1)
{
    assert(image_size != NULL);
    assert(feature_map_size != NULL);
    assert(this_scale != NULL);
    assert(next_scale != NULL);
    
    prior_box *box = (prior_box *)malloc(sizeof(prior_box));
    /*
    box->image_size = *image_size;
    box->feature_map_size = *feature_map_size;
    box->this_scale = *this_scale;
    box->next_scale = *next_scale;
    box->normalize_coords = (normalize_coords==NULL)?false:normalize_coords[0];
    box->clip_boxes = (clip_boxes==NULL)?false:clip_boxes[0];
     */
    pb_ratios ratios;
    if (ratios1 == NULL) {
        ratios.aspect_ratios[0] = 0.5f;
        ratios.aspect_ratios[1] = 0.7f;
        ratios.aspect_ratios[2] = 1.0f;
        ratios.aspect_ratios[3] = 1.4f;
        ratios.aspect_ratios[4] = 2.0f;
        ratios.count = 5;
//        ratios.two_boxes_for_ar1 = true;
    }
    else {
        ratios = *ratios1;
//        box->ratios.count = ratios->count;
//        npmemcpy(box->ratios.aspect_ratios, ratios->aspect_ratios, ratios->count*sizeof(float));
//        box->ratios.two_boxes_for_ar1 = ratios->two_boxes_for_ar1;
    }
    
    pb_vector steps;
    if (steps1 == NULL) {
        steps.dx = image_size->width/feature_map_size->width;
        steps.dy = image_size->height/feature_map_size->height;
    }
    else {
        steps.dx = steps1->dx;
        steps.dy = steps1->dy;
    }
    
    pb_vector offsets;
    if (offsets1 == NULL) {
        offsets.dx = 0.5f;
        offsets.dy = 0.5f;
    }
    else {
        offsets.dx = offsets1->dx;
        offsets.dy = offsets1->dy;
    }

    box->n_ars = number_of_ars(&ratios, two_boxes_for_ar1);
    assert(box->n_ars>0);
    pb_size *box_sizes = malloc(box->n_ars * sizeof(pb_size));
    make_pb_sizes(box_sizes, &box->n_ars, &ratios, two_boxes_for_ar1, image_size, this_scale, next_scale);
    
    box->meshgrid.cx = malloc(feature_map_size->width * sizeof(float));
    box->meshgrid.cy = malloc(feature_map_size->height * sizeof(float));
    box->meshgrid.nx = feature_map_size->width;
    box->meshgrid.ny = feature_map_size->height;
    make_pb_meshgrid(&box->meshgrid, &offsets, &steps);
    
    box->n_boxes = box->n_ars*feature_map_size->width*feature_map_size->height;
    assert(box->n_boxes>0);
    box->boxes = malloc(box->n_boxes*sizeof(pb_rect));
    make_prior_boxes(box->boxes, &box->meshgrid, box->n_ars, box_sizes, coords);
    free(box_sizes);
    
    if (clip_boxes) {
        for (int i = 0; i < box->n_boxes; i++) {
            clip_box(&box->boxes[i], image_size, coords);
        }
    }
    
    if (normalize_coords) {
        for (int i = 0; i < box->n_boxes; i++) {
            normalize_box(&box->boxes[i], image_size, coords);
        }
    }
    
    return box;
}

void prior_box_destroy(prior_box *box)
{
    if (box->boxes != NULL) {
        free(box->boxes);
    }
//    if (box->box_sizes) {
//        free(box->box_sizes);
//    }
    if (box->meshgrid.cx != NULL) {
        free(box->meshgrid.cx);
        free(box->meshgrid.cy);
    }
    free(box);
}

int number_of_ars(pb_ratios *aspect_ratios, bool two_boxes_for_ar1)
{
    int n_ars = aspect_ratios->count;
    if (two_boxes_for_ar1) {
        for (int i = 0; i < aspect_ratios->count; i++) {
            if (aspect_ratios->aspect_ratios[i] == 1.0f) {
                n_ars ++;
                break;
            }
        }
    }
    return n_ars;
}

void make_pb_sizes(pb_size *sizes, int *n_ars, pb_ratios *aspect_ratios, bool two_boxes_for_ar1,
                   pb_size *image_size, float *this_scale, float *next_scale)
{
    assert(*this_scale > 0.0f);
    assert(*next_scale > 0.0f);
    float s = fminf(image_size->width, image_size->height);
    int double_ar1 = 0;
    for (int i = 0; i < aspect_ratios->count; i++) {
        float ar = aspect_ratios->aspect_ratios[i];
        assert(ar > 0.0f);
        if (ar == 1.0f) {
            sizes[i].width = sizes[i].height = s**this_scale;
            if (two_boxes_for_ar1) {
                sizes[i+1].width = sizes[i+1].height = sqrtf(*this_scale**next_scale)*s;
                double_ar1 = 1;
            }
        }
        else {
            sizes[i+double_ar1].width = s**this_scale*sqrtf(ar);
            sizes[i+double_ar1].height = s**this_scale/sqrtf(ar);
        }
    }
}

void make_pb_meshgrid(pb_meshgrid *meshgrid, pb_vector *offsets, pb_vector *steps)
{
    linspace(offsets->dx*steps->dx, (offsets->dx+meshgrid->nx-1)*steps->dx, meshgrid->nx, true, meshgrid->cx, NULL);
    linspace(offsets->dy*steps->dy, (offsets->dy+meshgrid->ny-1)*steps->dy, meshgrid->ny, true, meshgrid->cy, NULL);
}

void make_prior_boxes(pb_rect *boxes, pb_meshgrid *meshgrid, int n_ars, pb_size *sizes, char coords)
{
    float cx, cy;
    float xmin, ymin, xmax, ymax;
    int a = 0;
    for (int i=0; i<meshgrid->ny; i++) {
        cy = meshgrid->cy[i];
        for (int j=0; j<meshgrid->nx; j++) {
            cx = meshgrid->cx[j];
            for (int k=0; k<n_ars; k++) {
                if (coords == 'm') {
                    ssd_centroids_to_minmax(&xmin, &xmax, &ymin, &ymax, cx, cy, sizes[k].width, sizes[k].height);
                    boxes[a].coords[0] = xmin;
                    boxes[a].coords[1] = xmax;
                    boxes[a].coords[2] = ymin;
                    boxes[a].coords[3] = ymax;
                }
                else if (coords == 'c') {
                    boxes[a].coords[0] = cx;
                    boxes[a].coords[1] = cy;
                    boxes[a].coords[2] = sizes[k].width;
                    boxes[a].coords[3] = sizes[k].height;
                }
                else {
                    assert(false);
                }
                a++;
            }
        }
    }
}

void normalize_box(pb_rect *rect, pb_size *image_size, char coords)
{
    if (coords == 'm') {
        rect->coords[0] /= image_size->width;
        rect->coords[1] /= image_size->width;
        rect->coords[2] /= image_size->height;
        rect->coords[3] /= image_size->height;
    }
    else if (coords == 'c') {
        rect->coords[0] /= image_size->width;
        rect->coords[1] /= image_size->height;
        rect->coords[2] /= image_size->width;
        rect->coords[3] /= image_size->height;
    }
    else {
        assert(false);
    }
}

void clip_box(pb_rect *rect, pb_size *image_size, char coords)
{
    if (coords == 'm') {
        rect->coords[0] = fmaxf(rect->coords[0], 0.0f);
        rect->coords[1] = fminf(rect->coords[1], image_size->width);
        rect->coords[2] = fmaxf(rect->coords[2], 0.0f);
        rect->coords[3] = fminf(rect->coords[3], image_size->height);
    }
    else if (coords == 'c') {
        float xmin, ymin, xmax, ymax;
        ssd_centroids_to_minmax(&xmin, &xmax, &ymin, &ymax, rect->coords[0], rect->coords[1], rect->coords[2], rect->coords[3]);
        xmin = fmaxf(xmin, 0.0f);
        xmax = fminf(xmax, image_size->width);
        ymin = fmaxf(ymin, 0.0f);
        ymax = fminf(ymax, image_size->height);
        ssd_minmax_to_centroids(&rect->coords[0], &rect->coords[1], &rect->coords[2], &rect->coords[3], xmin, xmax, ymin, ymax);
    }
    else {
        assert(false);
    }
}

void log_the_boxes(prior_box *box)
{
    printf("\n=======================================");
    printf("\n             %d x %d x %d              ", box->meshgrid.ny, box->meshgrid.nx, box->n_ars);
    printf("\n---------------------------------------");
    int a = 0;
    for (int i = 0; i < box->meshgrid.ny; i++) {
        for (int j = 0; j < box->meshgrid.nx; j++) {
            for (int k = 0; k < box->n_ars; k++) {
                pb_rect *rect = &box->boxes[a];
                printf("\n [%0.5f, %0.5f, %0.5f, %0.5f]   ", rect->coords[0], rect->coords[2], rect->coords[1], rect->coords[3]);
                a ++;
            }
            printf("\n");
        }
        printf("\n\n");
    }
    printf("\n---------------------------------------");
    printf("\n           total boxes: %d             ", box->n_boxes);
    printf("\n=======================================\n");
}

ssd_prior_box *ssd_prior_box_create(pb_size *image_size, int n_pyramid_layers, pb_size *feature_map_sizes,
                                    float *scales, pb_ratios *ratios, char coords,
                                    bool two_boxes_for_ar1, bool normalize_coords, bool clip_boxes,
                                    pb_vector *steps, pb_vector *offsets)
{
    assert(image_size);
    assert(feature_map_sizes);
    assert(n_pyramid_layers > 0);
    assert(scales);
    
    ssd_prior_box *box = malloc(sizeof(ssd_prior_box));
    box->image_size = *image_size;
    box->n_pyramid_layres = n_pyramid_layers;
    box->normalize_coords = normalize_coords;
    box->clip_boxes = clip_boxes;
    
    box->feature_map_sizes = malloc(n_pyramid_layers*sizeof(pb_size));
    npmemcpy(box->feature_map_sizes, feature_map_sizes, n_pyramid_layers*sizeof(pb_size));
    
    box->scales = malloc((n_pyramid_layers+1)*sizeof(float));
    npmemcpy(box->scales, scales, (n_pyramid_layers+1)*sizeof(float));
    
    box->ratios = malloc(n_pyramid_layers*sizeof(pb_ratios));
    npmemcpy(box->ratios, ratios, n_pyramid_layers*sizeof(pb_ratios));
    
    if (steps != NULL) {
        box->steps = malloc(n_pyramid_layers*sizeof(pb_vector));
        npmemcpy(box->steps, steps, n_pyramid_layers*sizeof(pb_vector));
    }
    else {
        box->steps = NULL;
    }
    
    if (offsets != NULL) {
        box->offsets = malloc(n_pyramid_layers*sizeof(pb_vector));
        npmemcpy(box->offsets, offsets, n_pyramid_layers*sizeof(pb_vector));
    }
    else {
        box->offsets = NULL;
    }
    
    box->n_boxes = 0;
    box->layers = malloc(n_pyramid_layers * sizeof(prior_box *));
    pb_vector *step = NULL;
    pb_vector *offset = NULL;
    for (int i = 0; i < n_pyramid_layers; i++) {
        if (steps) {
            step = &steps[i];
        }
        if (offsets) {
            offset = &offsets[i];
        }
        box->layers[i] = prior_box_create(image_size, &feature_map_sizes[i], coords, &scales[i], &scales[i+1],
                                          &ratios[i], two_boxes_for_ar1, normalize_coords, clip_boxes, step, offset);
        box->n_boxes += box->layers[i]->n_boxes;
    }
    
    return box;
}

void ssd_prior_box_destroy(ssd_prior_box *box)
{
    free(box->feature_map_sizes);
    free(box->scales);
    free(box->ratios);
    if (box->steps != NULL) {
        free(box->steps);
    }
    if (box->offsets != NULL) {
        free(box->offsets);
    }
    for (int i = 0; i < box->n_pyramid_layres; i++) {
        prior_box_destroy(box->layers[i]);
    }
    free(box);
}


void ssd_prior_box_get_all_boxes(pb_rect *dst, ssd_prior_box *buffer)
{
    int offset = 0;
    for (int i = 0; i < buffer->n_pyramid_layres; i++) {
        prior_box *layer = buffer->layers[i];
        npmemcpy(dst+offset, layer->boxes, layer->n_boxes*sizeof(pb_rect));
        offset+=layer->n_boxes;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
// converting coords between minmax and centroids

void ssd_minmax_to_centroids(float *cx, float *cy, float *w, float *h,
                             float xmin, float xmax, float ymin, float ymax)
{
    assert(xmax>xmin);
    assert(ymax>ymin);
    *cx = (xmin+xmax)*0.5f;
    *cy = (ymin+ymax)*0.5f;
    *w = xmax-xmin;
    *h = ymax-ymin;
}

void ssd_centroids_to_minmax(float *xmin, float *xmax, float *ymin, float *ymax,
                             float cx, float cy, float w, float h)
{
    *xmin = cx-w*0.5f;
    *xmax = cx+w*0.5f;
    *ymin = cy-h*0.5f;
    *ymax = cy+h*0.5f;
    assert(*xmax>*xmin);
    assert(*ymax>*ymin);
}

