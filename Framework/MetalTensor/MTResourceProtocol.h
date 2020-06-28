//
//  MTResourceProtocol.h
//  MetalTensor
//
//  Created by Feng Stone on 2020/2/21.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MPSImage+Extension.h"
#import "metal_tensor_structures.h"

NS_ASSUME_NONNULL_BEGIN

@protocol MTResource <NSObject>

@required

@property (nonatomic, assign) NSInteger poolIdentifier;

@property (nonatomic, assign) BOOL referenceCountingEnable;
@property (nonatomic, readonly) int referenceCounting;

- (void)lock;
- (void)unlock;

- (void)newContentOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
- (void)deleteContent;

- (NSString *)reuseIdentifier;

@end

NS_ASSUME_NONNULL_END
