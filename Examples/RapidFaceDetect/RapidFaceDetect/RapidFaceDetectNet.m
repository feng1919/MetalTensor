//
//  RapidFaceDetectNet.m
//  RapidFaceDetectNet
//
//  Created by Feng Stone on 2019/11/13.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "RapidFaceDetectNet.h"
#import <MetalTensor/ssd_decoder.h>
#import <MetalTensor/MetalTensorOutputLayer.h>
#import <MetalTensor/SSDConfig.h>

@interface RapidFaceDetectNet ()
{
    // outputs of model
    MetalTensorOutputLayer *locationOutput;
    MetalTensorOutputLayer *confidenceOutput;
    
    // decoder
    ssd_decoder *decoder;
    
    // configurations of SSD
    SSDConfig *config;
}

@end

@implementation RapidFaceDetectNet


- (instancetype)init {
    NSString *infoPlist = [[NSBundle mainBundle] pathForResource:@"RapidFaceDetect" ofType:@"plist"];
    NSString *configPlist = [[NSBundle mainBundle] pathForResource:@"RapidFaceDetectConfig" ofType:@"plist"];
    return [self initWithNetInfo:infoPlist config:configPlist];
}

- (instancetype)initWithNetInfo:(NSString *)infoPlist config:(NSString *)configPlist {
    if (self = [super initWithDictionary:[NSDictionary dictionaryWithContentsOfFile:infoPlist removingBlankSpaces:YES]]) {
        config = [[SSDConfig alloc] initWithDictionary:[NSDictionary dictionaryWithContentsOfFile:configPlist removingBlankSpaces:YES]];
    }
    return self;
}

- (void)dealloc {
    if (decoder != NULL) {
        ssd_decoder_destroy(decoder);
        decoder = NULL;
    }
}

- (void)loadWeights {
    NSString *dataFile = [[NSBundle mainBundle] pathForResource:@"RapidFaceDetect" ofType:@"bin"];
    NSString *mapFile = [[NSBundle mainBundle] pathForResource:@"RapidFaceDetect" ofType:@"json"];
    [self loadWeights:dataFile mapFile:mapFile];
}

- (void)compile:(id<MTLDevice>)device {
    
    [super compile:device];
    
#if DEBUG
    self.verbose = 0;
#endif
    
    locationOutput = (MetalTensorOutputLayer *)[self layerWithName:@"output1000"];
    NSParameterAssert(locationOutput);
    DataShape locationShape = [locationOutput outputShape];
    
    confidenceOutput = (MetalTensorOutputLayer *)[self layerWithName:@"output2000"];
    NSParameterAssert(confidenceOutput);
    DataShape confidenceShape = [confidenceOutput outputShape];
    
    ssd_prior_box *box_buffer = ssd_prior_box_create([config networkSizeRef],
                                                     config.n_pyramid_layers,
                                                     config.feature_map_sizes,
                                                     config.scales,
                                                     config.aspect_ratios,
                                                     config.coords,
                                                     config.two_boxes_for_ar1,
                                                     config.normalize_coords,
                                                     config.clip_boxes,
                                                     config.steps,
                                                     config.offsets);
    decoder = ssd_decoder_create(&locationShape, &confidenceShape, (int)config.classes.count, NULL);
    decoder->confidence_thresh = config.confidence_thresh;
    decoder->iou_thresh = config.iou_thresh;
    decoder->coords = config.coords;
    npmemcpy(decoder->variances, [config variances], 4*sizeof(float));
    ssd_prior_box_get_all_boxes(decoder->boxes, box_buffer);
    ssd_prior_box_destroy(box_buffer);
    
    __weak __auto_type weakSelf = self;
    self.scheduledHandler = ^(id<MTLCommandBuffer> _Nonnull cmd) {
        [[FPSCounter sharedCounter] start];
    };
    self.completedHandler = ^(id<MTLCommandBuffer> _Nonnull cmd) {
            
        __strong __auto_type strongSelf = weakSelf;
        
        [[FPSCounter sharedCounter] stop];
                
        {
            DataShape *shape = [strongSelf->locationOutput outputShapeRef];
            int size = ProductOfDataShape(shape);
            float16_t *result = malloc(size*sizeof(float16_t));
            MPSImage *outputImage = [strongSelf->locationOutput outputImage];
            [outputImage toFloat16Array:result];
            float32_t *result32 = malloc(size*sizeof(float32_t));
            ConvertFloat16To32(result, result32, size);
            free(result);
            int offset0 = 0;
            int offset1 = 0;
            for (int i = 0; i < strongSelf->config.n_pyramid_layers; i++) {
                DataShape shape_i = strongSelf->config.location_shapes[i];
                ConvertToTensorFlowLayout(strongSelf->decoder->location+offset0, result32+offset1, &shape_i);
                offset0 += ProductOfDataShape(&shape_i);
                offset1 += ProductOfDataShapeDepth4Divisible(&shape_i);
            }
            
            free(result32);
        }
                
        {
            DataShape *shape = [strongSelf->confidenceOutput outputShapeRef];
            int size = ProductOfDataShape(shape);
            float16_t *result = malloc(size*sizeof(float16_t));
            MPSImage *outputImage = [strongSelf->confidenceOutput outputImage];
            [outputImage toFloat16Array:result];
            float32_t *result32 = malloc(size*sizeof(float32_t));
            ConvertFloat16To32(result, result32, size);
            free(result);
            int offset0 = 0;
            int offset1 = 0;
            for (int i = 0; i < strongSelf->config.n_pyramid_layers; i++) {
                DataShape shape_i = strongSelf->config.confidence_shapes[i];
                ConvertToTensorFlowLayout(strongSelf->decoder->confidence+offset0, result32+offset1, &shape_i);
                offset0 += ProductOfDataShape(&shape_i);
                offset1 += ProductOfDataShapeDepth4Divisible(&shape_i);
            }
            
            free(result32);
        }
                
        ssd_decoder_process(strongSelf->decoder);
//        ssd_decoder_print_results(strongSelf->decoder);
        [strongSelf notifyResults];
    };
}

- (void)notifyResults {
    
    NSMutableArray *objects = [NSMutableArray array];
    for (int i = 0; i < decoder->n_objects; i++) {
        NSString *name = config.classes[decoder->objects[i].class_id];
        [objects addObject:[[SSDObject alloc] initWithSSDObject:&decoder->objects[i] name:name]];
    }
    self.objects = objects;
    
    [_delegate RapidFaceDetectNet:self didFinishWithObjects:objects];
}

@end
