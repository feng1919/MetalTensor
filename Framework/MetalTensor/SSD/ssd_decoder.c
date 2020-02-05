//
//  ssd_decoder.c
//  MetalImage
//
//  Created by Feng Stone on 2019/6/26.
//  Copyright © 2019 fengshi. All rights reserved.
//

#include "ssd_decoder.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include "numpy.h"

void ssd_decoder_sort_objects(ssd_object **objects,int begin, int end);
void ssd_decoder_greedy_nms(ssd_object **object, int count, float thresh, int *n_objects);
void decode_minmax_coordinates(ssd_decoding_unit *unit, float *variances, float *xmin, float *xmax, float *ymin, float *ymax);
void decode_centroids_coordinates(ssd_decoding_unit *unit, float *variances, float *xmin, float *xmax, float *ymin, float *ymax);

ssd_decoder *ssd_decoder_create(DataShape *location_shape, DataShape *confidence_shape, int n_classes, pb_size *image_size)
{
    assert(location_shape);
    assert(confidence_shape);
    assert(location_shape->depth==confidence_shape->depth);
    assert(location_shape->depth>0);
    assert(Product(location_shape)>0);
    assert(Product(confidence_shape)>0);
    assert(n_classes>1);
    
    ssd_decoder *decoder = malloc(sizeof(ssd_decoder));
    decoder->location_shape = *location_shape;
    decoder->confidence_shape = *confidence_shape;
    decoder->n_classes = n_classes;
    decoder->n_boxes = location_shape->depth;
    decoder->confidence_thresh = 0.5f;
    decoder->iou_thresh = 0.1f;
    decoder->normalize_coords = true;
    decoder->image_width = image_size?image_size->width:1.0f;
    decoder->image_height = image_size?image_size->height:1.0f;
    decoder->location = malloc(Product(location_shape)*sizeof(float));
    decoder->confidence = malloc(Product(confidence_shape)*sizeof(float));
    decoder->boxes = malloc(decoder->n_boxes*sizeof(pb_rect));
    decoder->n_objects = 0;
    decoder->objects = NULL;
    decoder->coords = 'm';
    
    return decoder;
}

float *ssd_decoder_get_location_buffer(ssd_decoder *decoder, int offset) {
    return decoder->location+offset;
}

float *ssd_decoder_get_confidence_buffer(ssd_decoder *decoder, int offset) {
    return decoder->confidence+offset;
}

void ssd_decoder_create_boxes(ssd_decoder *decoder, pb_size *network_image_size, int n_pyramid_layers, pb_size *feature_map_size,
                              float *scales, pb_ratios *ratios, bool two_boxes_for_ar1, bool normalize_coords, bool clip_boxes,
                              pb_vector *steps, pb_vector *offsets)
{
    ssd_prior_box *box_buffer = ssd_prior_box_create(network_image_size, n_pyramid_layers, feature_map_size, scales, ratios,
                                                     decoder->coords, two_boxes_for_ar1, normalize_coords, clip_boxes, steps, offsets);
    if (decoder->boxes != NULL) {
        free(decoder->boxes);
    }
    decoder->boxes = malloc(box_buffer->n_boxes*sizeof(pb_rect));
    ssd_prior_box_get_all_boxes(decoder->boxes, box_buffer);
    ssd_prior_box_destroy(box_buffer);
}

void ssd_decoder_destroy(ssd_decoder *decoder) {
    free(decoder->location);
    free(decoder->confidence);
    if (decoder->boxes != NULL) {
        free(decoder->boxes);
    }
    free(decoder);
}

void ssd_decoder_process(ssd_decoder *decoder) {
    int n_unit_max = decoder->confidence_shape.depth;
    int n_classes = decoder->n_classes;
    int n_unit = 0;
    ssd_decoding_unit *unit_list = malloc(n_unit_max*sizeof(ssd_decoding_unit));
    for (int i = 0; i < n_unit_max; i++) {
        // Since the most of predictions are background objects,
        // we remove them firstly, because of the cost of exp computation.
        if (argmax(decoder->confidence+i*n_classes, n_classes) > 0) {
            unit_list[n_unit].location = decoder->location+i*4;
            unit_list[n_unit].confidence = decoder->confidence+i*n_classes;
            unit_list[n_unit].box_rect = &decoder->boxes[i];
            n_unit++;
        }
    }
    
    // If all objects were background objects.
    if (n_unit == 0) {
        free(unit_list);
        goto NO_OBJECT_MATCH;
    }
    
    // 1. Cast softmax on the confidences.
    // 2. Obtain the rate of the classes.
    for (int i = 0; i < n_unit; i++) {
        soft_max(unit_list[i].confidence, n_classes);
    }
    
    // Generate the class objects.
    ssd_object *objects = malloc(n_unit*sizeof(ssd_object));
    int n_conf_matches = 0;
    for (int i = 0; i < n_unit; i++) {
        int class_id = argmax(unit_list[i].confidence, n_classes);
        float confidence = unit_list[i].confidence[class_id];
        if (confidence >= decoder->confidence_thresh) {
            objects[n_conf_matches].class_id = class_id;
            objects[n_conf_matches].confidence = confidence;
            // Decoding the location coordinates.
            if (decoder->coords == 'm') {
                decode_minmax_coordinates(&unit_list[i], decoder->variances, &objects[n_conf_matches].xmin, &objects[n_conf_matches].xmax, &objects[n_conf_matches].ymin, &objects[n_conf_matches].ymax);
            }
            else if (decoder->coords == 'c') {
                decode_centroids_coordinates(&unit_list[i], decoder->variances, &objects[n_conf_matches].xmin, &objects[n_conf_matches].xmax, &objects[n_conf_matches].ymin, &objects[n_conf_matches].ymax);
            }
            else {
                assert(false);
            }
            n_conf_matches++;
        }
    }
    free(unit_list);
    
    // If there were no object match the confidence.
    if (n_conf_matches == 0) {
        free(objects);
        goto NO_OBJECT_MATCH;
    }
    
    // Sort the objects by confidences.
    ssd_object **object_ref_list = malloc(n_conf_matches*sizeof(ssd_object *));
    for (int i = 0; i < n_conf_matches; i++) {
        object_ref_list[i] = &objects[i];
    }
    ssd_decoder_sort_objects(object_ref_list, 0, n_conf_matches-1);
    
    // Perform greedy non-maximum suppression on the results.
    ssd_decoder_greedy_nms(object_ref_list, n_conf_matches, decoder->iou_thresh, &decoder->n_objects);
    assert(decoder->n_objects > 0);
    
    // Output the results.
    if (decoder->objects != NULL) {
        free(decoder->objects);
    }
    decoder->objects = malloc(decoder->n_objects*sizeof(ssd_object));
    int a = 0;
    for (int i = 0; i < n_conf_matches; i++) {
        ssd_object *obj = object_ref_list[i];
        if (obj->confidence > decoder->confidence_thresh) {
            npmemcpy(&decoder->objects[a], obj, sizeof(ssd_object));
            a++;
        }
    }
    assert(a == decoder->n_objects);
    
    free(object_ref_list);
    free(objects);
    
    return;
    
NO_OBJECT_MATCH:
    if (decoder->objects != NULL) {
        free(decoder->objects);
        decoder->objects = NULL;
    }
    decoder->n_objects = 0;
    return;
}

void ssd_decoder_sort_objects(ssd_object **objects,int begin, int end)
{
    if (begin >= end) {
        return;
    }
    
    int i = begin;
    int j = end;
    int k = i;
    ssd_object *temp = NULL;
    
    while (i < j) {
        ssd_object *ok = objects[k];
        if (k == i) {
            ssd_object *oj = objects[j];
            if (ok->confidence < oj->confidence) {
                temp = objects[k];
                objects[k] = oj;
                objects[j] = temp;
                k = j;
                i++;
            }
            else {
                j--;
            }
        }
        else {
            ssd_object *oi = objects[i];
            if (oi->confidence < ok->confidence) {
                temp = objects[k];
                objects[k] = oi;
                objects[i] = temp;
                k = i;
                j--;
            }
            else {
                i ++;
            }
        }
    }
    ssd_decoder_sort_objects(objects, begin, k-1);
    ssd_decoder_sort_objects(objects, k+1, end);
}

void ssd_decoder_greedy_nms(ssd_object **objects, int count, float iou_thresh, int *n_objects)
{
    int n_neg_objects = 0;
    for (int i = 0; i < count; i++) {
        ssd_object *obj = objects[i];
        if (obj->confidence < 0.01) {
            continue;
        }
        
        for (int j = i+1; j < count; j++) {
            ssd_object *obj1 = objects[j];
            if (obj1->confidence < 0.01) {
                continue;
            }
            
            if (obj->xmin>=obj1->xmax || obj->xmax<=obj1->xmin ||
                obj->ymin>=obj1->ymax || obj->ymax<=obj1->ymin) {
                continue;
            }
            float width = (obj->xmax>obj1->xmin)?(obj->xmax-obj1->xmin):(obj1->xmax-obj->xmin);
            float height = (obj->ymax>obj1->ymin)?(obj->ymax-obj1->ymin):(obj1->ymax-obj->ymin);
            float s0 = width*height;
            float s1 = (obj->xmax-obj->xmin)*(obj->ymax-obj->ymin)+(obj1->xmax-obj1->xmin)*(obj1->ymax-obj1->ymin)-s0;
            if (s0/s1 > iou_thresh) {
                obj1->confidence = 0.0f;
                n_neg_objects++;
            }
        }
    }
    n_objects[0] = count-n_neg_objects;
}

void decode_minmax_coordinates(ssd_decoding_unit *unit, float *variances, float *xmin, float *xmax, float *ymin, float *ymax)
{
    // 1.
    // y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:]
    // delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor)
    // for all four coordinates, where 'size' refers to w or h, respectively
 
    // 2.
    // y_pred_decoded_raw[:,:,[-4,-3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1)
    // delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred)
    // delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)

    // 3.
    // y_pred_decoded_raw[:,:,[-2,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1)
    // delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred)
    // delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)

    // 4.
    // y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4]
    // delta(pred) + anchor == pred for all four coordinates
    
    pb_rect *rect = unit->box_rect;
    *xmin = unit->location[0]*variances[0]*(rect->coords[1]-rect->coords[0])+rect->coords[0];
    *xmax = unit->location[1]*variances[1]*(rect->coords[1]-rect->coords[0])+rect->coords[1];
    *ymin = unit->location[2]*variances[2]*(rect->coords[3]-rect->coords[2])+rect->coords[2];
    *ymax = unit->location[3]*variances[3]*(rect->coords[3]-rect->coords[2])+rect->coords[3];
}

void decode_centroids_coordinates(ssd_decoding_unit *unit, float *variances, float *xmin, float *xmax, float *ymin, float *ymax)
{
    // 1.
    // y_pred_converted[:,:,[4,5]] = np.exp(y_pred_converted[:,:,[4,5]] * y_pred[:,:,[-2,-1]])
    // exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
    
    // 2.
    // y_pred_converted[:,:,[4,5]] *= y_pred[:,:,[-6,-5]]
    // (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
    
    // 3.
    // y_pred_converted[:,:,[2,3]] *= y_pred[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]]
    // (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
    
    // 4.
    // y_pred_converted[:,:,[2,3]] += y_pred[:,:,[-8,-7]]
    // delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
    
    // 5.
    // y_pred_converted = convert_coordinates(y_pred_converted, start_index=-4, conversion='centroids2corners')
    
    float cx, cy, w, h;
    pb_rect *rect = unit->box_rect;
    cx = unit->location[0]*rect->coords[2]*variances[0]+rect->coords[0];
    cy = unit->location[1]*rect->coords[3]*variances[1]+rect->coords[1];
    w = expf(unit->location[2]*variances[2])*rect->coords[2];
    h = expf(unit->location[3]*variances[3])*rect->coords[3];
    ssd_centroids_to_minmax(xmin, xmax, ymin, ymax, cx, cy, w, h);
}

void ssd_decoder_print_results(ssd_decoder *decoder) {
    printf("\n=============================================");
    printf("\n                 RESULT: %d                  ", decoder->n_objects);
    printf("\n---------------------------------------------");
    printf("\n             IMAGE SIZE: (%0.f, %0.f)        ", decoder->image_width, decoder->image_height);
    printf("\n---------------------------------------------");
    float w = decoder->image_width;
    float h = decoder->image_height;
    for (int i = 0; i < decoder->n_objects; i++) {
        printf("\n [%d, %0.2f, (%0.1f, %0.1f, %0.1f, %0.1f)]", decoder->objects[i].class_id, decoder->objects[i].confidence,
               decoder->objects[i].xmin*w, decoder->objects[i].xmax*w, decoder->objects[i].ymin*h, decoder->objects[i].ymax*h);
    }
    printf("\n=============================================\n");
}
