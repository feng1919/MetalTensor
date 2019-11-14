//
//  RapidFaceDetectFilter.h
//  RapidFaceDetect
//
//  Created by Feng Stone on 2019/11/13.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <MetalImage/MetalImage.h>

NS_ASSUME_NONNULL_BEGIN

@interface RapidFaceDetectFilter : MetalImageFilter

- (void)createNet;

@end

NS_ASSUME_NONNULL_END
