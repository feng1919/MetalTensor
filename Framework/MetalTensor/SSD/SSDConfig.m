//
//  SSDConfig.m
//  MetalImage
//
//  Created by Feng Stone on 2019/7/1.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "SSDConfig.h"
#import "NSString+Extension.h"

@implementation SSDConfig

- (instancetype)initWithDictionary:(NSDictionary *)dictionary {
    if (self = [super init]) {
        
        self.classes = [dictionary[@"classes"] nonEmptyComponentsSeparatedByString:@","];
        
        NSString *coords = dictionary[@"coords"]?:@"minmax";
        self.coords = [coords substringToIndex:1].UTF8String[0];
        
        NSParameterAssert(dictionary[@"network_size"]);
        NSArray<NSString *> *networkSizeInfo = [dictionary[@"network_size"] nonEmptyComponentsSeparatedByString:@","];
        NSParameterAssert(networkSizeInfo.count == 2);
        _network_size.width = [networkSizeInfo[0] floatValue];
        _network_size.height = [networkSizeInfo[1] floatValue];
        
        NSParameterAssert(dictionary[@"n_pyramid_layers"]);
        _n_pyramid_layers = [dictionary[@"n_pyramid_layers"] intValue];
        
        NSParameterAssert(dictionary[@"feature_map_sizes"]);
        NSArray<NSString *> *feature_map_sizes = dictionary[@"feature_map_sizes"];
        _feature_map_sizes = malloc(_n_pyramid_layers*sizeof(pb_size));
        NSParameterAssert(feature_map_sizes.count == _n_pyramid_layers);
        for (int i = 0; i < _n_pyramid_layers; i++) {
            NSArray<NSString *> *sizeInfo = [feature_map_sizes[i] nonEmptyComponentsSeparatedByString:@","];
            NSParameterAssert(sizeInfo.count == 2);
            _feature_map_sizes[i].width = [sizeInfo[0] floatValue];
            _feature_map_sizes[i].height = [sizeInfo[1] floatValue];
        }
        
        NSParameterAssert(dictionary[@"scales"]);
        NSArray<NSString *> *scales = [dictionary[@"scales"] nonEmptyComponentsSeparatedByString:@","];
        NSParameterAssert(scales.count == _n_pyramid_layers+1);
        _scales = malloc(scales.count*sizeof(float));
        for (int i = 0; i < scales.count; i++) {
            _scales[i] = [scales[i] floatValue];
        }
        
        NSParameterAssert(dictionary[@"aspect_ratios"]);
        NSArray<NSString *> *aspectRatiosList = dictionary[@"aspect_ratios"];
        NSParameterAssert(aspectRatiosList.count == _n_pyramid_layers);
        _aspect_ratios = malloc(aspectRatiosList.count*sizeof(pb_ratios));
        for (int i = 0; i < aspectRatiosList.count; i++) {
            NSArray<NSString *> *arInfo = [aspectRatiosList[i] nonEmptyComponentsSeparatedByString:@","];
            NSParameterAssert(arInfo.count<=8);
            _aspect_ratios[i].count = (int)arInfo.count;
            for (int j = 0; j < arInfo.count; j++) {
                _aspect_ratios[i].aspect_ratios[j] = [arInfo[j] floatValue];
            }
        }
        
        _two_boxes_for_ar1 = dictionary[@"two_boxes_for_ar1"]?[dictionary[@"two_boxes_for_ar1"] boolValue]:true;
        _normalize_coords = dictionary[@"normalize_coords"]?[dictionary[@"normalize_coords"] boolValue]:true;
        _clip_boxes = dictionary[@"clip_boxes"]?[dictionary[@"clip_boxes"] boolValue]:false;
        _confidence_thresh = dictionary[@"confidence_thresh"]?[dictionary[@"confidence_thresh"] floatValue]:0.5f;
        _iou_thresh = dictionary[@"iou_thresh"]?[dictionary[@"iou_thresh"] floatValue]:0.1f;
        
        _variances = malloc(4*sizeof(float));
        NSParameterAssert(dictionary[@"variances"]);
        NSArray<NSString *> *variances = [dictionary[@"variances"] nonEmptyComponentsSeparatedByString:@","];
        NSParameterAssert(variances.count == 4);
        for (int i = 0; i < variances.count; i++) {
            _variances[i] = [variances[i] floatValue];
        }
        
        if (dictionary[@"steps"]) {
            NSArray<NSString *> *stepsList = dictionary[@"steps"];
            NSParameterAssert(stepsList.count == _n_pyramid_layers);
            _steps = malloc(stepsList.count*sizeof(pb_vector));
            for (int i = 0; i < stepsList.count; i++) {
                NSArray<NSString *> *stepInfo = [stepsList[i] nonEmptyComponentsSeparatedByString:@","];
                NSParameterAssert(stepInfo.count == 2);
                _steps[i].dx = [stepInfo[0] floatValue];
                _steps[i].dy = [stepInfo[1] floatValue];
            }
        }
        
        if (dictionary[@"offsets"]) {
            NSArray<NSString *> *offsetsList = dictionary[@"offsets"];
            NSParameterAssert(offsetsList.count == _n_pyramid_layers);
            _offsets = malloc(offsetsList.count*sizeof(pb_vector));
            for (int i = 0; i < offsetsList.count; i++) {
                NSArray<NSString *> *offsetsInfo = [offsetsList[i] nonEmptyComponentsSeparatedByString:@","];
                NSParameterAssert(offsetsInfo.count == 2);
                _offsets[i].dx = [offsetsInfo[0] floatValue];
                _offsets[i].dy = [offsetsInfo[1] floatValue];
            }
        }
        
        // Analyse the output shapes
        NSParameterAssert(dictionary[@"location_shapes"]);
        NSArray<NSString *> *locationList = [dictionary[@"location_shapes"] nonEmptyComponentsSeparatedByString:@";"];
        NSParameterAssert(locationList.count == _n_pyramid_layers);
        _location_shapes = malloc(_n_pyramid_layers * sizeof(DataShape));
        for (int i = 0; i < locationList.count; i++) {
            NSArray<NSString *> *inputInfo = [locationList[i] nonEmptyComponentsSeparatedByString:@","];
            NSAssert(inputInfo.count == 3, @"Invliad output shape: '%@'", dictionary[@"location_shapes"]);
            _location_shapes[i].row = [inputInfo[0] intValue];
            _location_shapes[i].column = [inputInfo[1] intValue];
            _location_shapes[i].depth = [inputInfo[2] intValue];
        }
        
        NSParameterAssert(dictionary[@"confidence_shapes"]);
        NSArray<NSString *> *confidenceList = [dictionary[@"confidence_shapes"] nonEmptyComponentsSeparatedByString:@";"];
        NSParameterAssert(confidenceList.count == _n_pyramid_layers);
        _confidence_shapes = malloc(_n_pyramid_layers * sizeof(DataShape));
        for (int i = 0; i < confidenceList.count; i++) {
            NSArray<NSString *> *inputInfo = [confidenceList[i] nonEmptyComponentsSeparatedByString:@","];
            NSAssert(inputInfo.count == 3, @"Invliad output shape: '%@'", dictionary[@"confidence_shapes"]);
            _confidence_shapes[i].row = [inputInfo[0] intValue];
            _confidence_shapes[i].column = [inputInfo[1] intValue];
            _confidence_shapes[i].depth = [inputInfo[2] intValue];
        }
    }
    return self;
}

- (void)dealloc {
    free(_feature_map_sizes);
    free(_scales);
    free(_aspect_ratios);
    free(_variances);
    if (_steps) {
        free(_steps);
    }
    if (_offsets) {
        free(_offsets);
    }
    free(_location_shapes);
    free(_confidence_shapes);
}

- (pb_size *)networkSizeRef {
    return &_network_size;
}

@end
