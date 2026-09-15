//
//  SSDConfig.h
//  MetalImage
//
//  Created by Feng Stone on 2019/7/1.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#include "ssd_prior_box.h"
#include "metal_tensor_structures.h"

NS_ASSUME_NONNULL_BEGIN

@interface SSDConfig : NSObject

@property (nonatomic, strong) NSArray<NSString *> *classes;
@property (nonatomic, assign) char coords;
@property (nonatomic, assign) pb_size network_size;
@property (nonatomic, assign) int n_pyramid_layers;
@property (nonatomic, assign) pb_size *feature_map_sizes;
@property (nonatomic, assign) float *scales;
@property (nonatomic, assign) float *variances;
@property (nonatomic, assign) pb_ratios *aspect_ratios;
@property (nonatomic, assign) bool two_boxes_for_ar1;
@property (nonatomic, assign) bool normalize_coords;
@property (nonatomic, assign) bool clip_boxes;
@property (nonatomic, assign) float confidence_thresh;
@property (nonatomic, assign) float iou_thresh;
@property (nonatomic, assign) pb_vector *steps;
@property (nonatomic, assign) pb_vector *offsets;
@property (nonatomic, assign) DataShape *location_shapes;
@property (nonatomic, assign) DataShape *confidence_shapes;

- (instancetype)initWithDictionary:(NSDictionary *)dictionary;

- (pb_size *)networkSizeRef;

@end

NS_ASSUME_NONNULL_END
