//
//  CropFilter.h
//  RapidFaceDetect
//
//  Created by Feng Stone on 2020/7/26.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import <MetalImage/MetalImage.h>

NS_ASSUME_NONNULL_BEGIN

@interface CropFilter : MICropFilter

@property (nonatomic, assign) BOOL framebuffer;

@end

NS_ASSUME_NONNULL_END
