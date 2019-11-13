//
//  SSDObject.h
//  MetalImage
//
//  Created by Feng Stone on 2019/7/1.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#include "ssd_decoder.h"

NS_ASSUME_NONNULL_BEGIN

@interface SSDObject : NSObject

@property (nonatomic, strong) NSString *name;
@property (nonatomic, assign) int confidence;
@property (nonatomic, assign) float xmin;
@property (nonatomic, assign) float xmax;
@property (nonatomic, assign) float ymin;
@property (nonatomic, assign) float ymax;

- (instancetype)initWithSSDObject:(ssd_object *)object name:(NSString *)name;

SSDObject *MakeSSDObject(ssd_object *object, NSString *name);

@end

NS_ASSUME_NONNULL_END
