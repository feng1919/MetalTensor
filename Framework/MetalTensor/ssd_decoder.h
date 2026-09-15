//
//  ssd_decoder.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/26.
//  Copyright © 2019 fengshi. All rights reserved.
//

#ifndef ssd_decoder_h
#define ssd_decoder_h

#include <stdio.h>
#include <stdbool.h>
#include "ssd_prior_box.h"
#include "metal_tensor_structures.h"

typedef struct ssd_object{
    int class_id;
    float confidence;
    float xmin;
    float xmax;
    float ymin;
    float ymax;
}ssd_object;

typedef struct ssd_decoding_unit{
    float *confidence;
    float *location;
    pb_rect *box_rect;
}ssd_decoding_unit;

typedef struct ssd_decoder{
    
    /*
     *  Configuration for decoder
     */
    float confidence_thresh; // 0.5
    float iou_thresh; // 0.1
    bool normalize_coords; // true
    float image_width; // 1.0f, if not set, normalized results will be return.
    float image_height; // 1.0f, if not set, normalized results will be return.
    float variances[4]; // (0.1f, 0.1f, 0.2f, 0.2f)
    char coords;        // 'c':centroids, 'm':minmax, default 'm'
    
    /*
     *  location buffers
     */
    DataShape location_shape;
    float *location;
    
    /*
     *  confidence buffers
     */
    DataShape confidence_shape;
    float *confidence;
    int n_classes; // including background
    
    /*
     *  prior boxes buffers
     */
    pb_rect *boxes;
    int n_boxes;
    
    /*
     *  the decoded objects
     */
    ssd_object *objects;
    int n_objects;
    
}ssd_decoder;

ssd_decoder *ssd_decoder_create(DataShape *location_shape, DataShape *confidence_shape, int n_classes, pb_size *image_size);
void ssd_decoder_create_boxes(ssd_decoder *decoder, pb_size *network_image_size, int n_pyramid_layers, pb_size *feature_map_size,
                              float *scales, pb_ratios *ratios, bool two_boxes_for_ar1, bool normalize_coords, bool clip_boxes,
                              pb_vector *steps, pb_vector *offsets);
float *ssd_decoder_get_location_buffer(ssd_decoder *decoder, int offset);
float *ssd_decoder_get_confidence_buffer(ssd_decoder *decoder, int offset);
void ssd_decoder_process(ssd_decoder *decoder);
void ssd_decoder_print_results(ssd_decoder *decoder);
void ssd_decoder_destroy(ssd_decoder *decoder);

float ssd_iou(ssd_object *obj1, ssd_object *obj2);

#endif /* ssd_decoder_h */
