//
//  LandmarksFilter.h
//  RapidFaceDetect
//
//  Created by Feng Stone on 2020/7/25.
//  Copyright © 2020 fengshi. All rights reserved.
//

#if __has_include(<MetalImage/MetalImage.h>)
#import <MetalImage/MetalImage.h>
#else
#import "MetalImage.h"
#endif

NS_ASSUME_NONNULL_BEGIN

@interface LandmarksFilter : MetalImageFilter

- (void)createNet;

@end

NS_ASSUME_NONNULL_END
