//
//  ImageNet1000.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/24.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface ImageNet1000 : NSObject

+ (id)sharedInstance;

- (float *)rateBuffer;
- (NSDictionary *)rank5;

@end

NS_ASSUME_NONNULL_END
