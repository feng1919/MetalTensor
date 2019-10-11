//
//  MetalTensorInput.h
//  MetalImage
//
//  Created by Feng Stone on 2019/5/20.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MITemporaryImage.h"

@protocol MetalTensorInput <NSObject>

@required
- (void)tempImageReadyAtIndex:(NSInteger)imageIndex commandBuffer:(id<MTLCommandBuffer>)cmdBuf;
- (void)setInputImage:(MITemporaryImage *)newInputImage atIndex:(NSInteger)imageIndex;
- (void)processTensorWithCommandBuffer:(id<MTLCommandBuffer>)cmdBuf;

@optional
- (BOOL)isAllReceived;
- (void)resetReceivedFlags;
- (NSInteger)nextAvailableTempImageIndex;
- (void)reserveTempImageIndex:(NSInteger)index;
- (void)releaseTempImageIndex:(NSInteger)index;

@end

@protocol MetalTensorWeights <NSObject>

@required
- (void)loadWeights;
- (BOOL)didLoadWeights;

@optional

- (void)loadWeights:(NSData *)weights
        kernelShape:(KernelShape *)kernelShape
         neuronType:(NeuronType *)neuronType
          depthWise:(BOOL)depthWise;

- (void)loadWeights:(NSString *)weights
              range:(NSRange *)range
        kernelShape:(KernelShape *)kernelShape
         neuronType:(NeuronType *)neuronType
          depthWise:(BOOL)depthWise;

- (void)loadWeightsList:(NSArray<NSString *> *)weightsList
              rangeList:(NSRange *)rangeList
           kernelShapes:(KernelShape *)kernelShapes
            neuronTypes:(NeuronType *)neuronTypes
             depthWises:(BOOL *)depthWises;

@end
