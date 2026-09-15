//
//  RapidFaceDetectFilter.h
//  RapidFaceDetect
//
//  Created by Feng Stone on 2019/11/13.
//  Copyright © 2019 fengshi. All rights reserved.
//

#if __has_include(<MetalImage/MetalImage.h>)
#import <MetalImage/MetalImage.h>
#else
#import "MetalImage.h"
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RapidFaceDetectFilter : MetalImageFilter

- (void)createNet;

@end

NS_ASSUME_NONNULL_END
