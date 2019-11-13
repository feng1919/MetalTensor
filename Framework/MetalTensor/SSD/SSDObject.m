//
//  SSDObject.m
//  MetalImage
//
//  Created by Feng Stone on 2019/7/1.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "SSDObject.h"

@implementation SSDObject

- (instancetype)initWithSSDObject:(ssd_object *)object name:(NSString *)name {
    if (self = [super init]) {
        self.name = name;
        self.confidence = object->confidence;
        self.xmin = object->xmin;
        self.xmax = object->xmax;
        self.ymin = object->ymin;
        self.ymax = object->ymax;
    }
    return self;
}

@end
