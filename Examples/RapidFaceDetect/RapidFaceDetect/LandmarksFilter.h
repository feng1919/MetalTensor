//
//  LandmarksFilter.h
//  RapidFaceDetect
//
//  Created by Feng Stone on 2020/7/25.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import <MetalImage/MetalImage.h>

NS_ASSUME_NONNULL_BEGIN

@interface LandmarksFilter : MetalImageFilter

- (void)createNet;

@end

NS_ASSUME_NONNULL_END
