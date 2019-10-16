//
//  MIConcatenateLayer.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/9.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalTensorLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MIConcatenateLayer : MetalTensorLayer <MetalTensorInput> {
    
@protected
    DataShape _tensorShape;
}

- (int *)channelOffsets;
- (DataShape *)tensorShape;

@end

NS_ASSUME_NONNULL_END
