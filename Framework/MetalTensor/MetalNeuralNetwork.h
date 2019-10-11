//
//  MetalNeuralNetwork.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/25.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MetalTensorInputLayer.h"
#import "MetalTensorOutputLayer.h"
#import "MetalTensorLayerDescriptor.h"
#import <MetalImage/MetalImageOutput.h>

NS_ASSUME_NONNULL_BEGIN

typedef void (^NetworkCallback)(id<MTLCommandBuffer>);

/*
 *  A MetalNeuralNetwork instance is a MetalImage node,
 *  which may connect to the MetalImage's rendering and
 *  parallel computation chain, and takes a MetalImageTexture
 *  as an input, also output to the targets.
 *
 */

@interface MetalNeuralNetwork : MetalImageOutput <MetalImageInput> {
    
@protected
    // NETWORK
    NSDictionary *_networkDesc;
    MetalTensorInputLayer *_inputLayer;
    NSMutableArray<MetalTensorOutputLayer *> *_outputLayers;
    NSMutableDictionary<NSString *, MetalTensorLayer *> *_allLayers;
    NSMutableDictionary<NSString *, MetalTensorLayerDescriptor *> *_allLayerDescriptors;
    NSString *_reuseIdentifier;
    
    // sync between network and MetalImage
    dispatch_semaphore_t  _network_semaphore;
    dispatch_queue_t _network_queue;
    
    MetalImageTexture *_inputTexture;
}

/*
 *  Callbacks for the custom tasks.
 */
@property (nonatomic, copy) NetworkCallback _Nullable scheduledHandler;
@property (nonatomic, copy) NetworkCallback _Nullable completedHandler;

/*
 *  The network's tasks should be processed synchronizely or asynchronizely.
 *  If sync the processing will be casted on the current thread,
 *  and if async the processing will be casted on the network thread.
 *
 *  It's sync by default.
 *
 */
@property (nonatomic, assign) BOOL synchronizedProcessing;

/*
 *
 *  ******************************************************
 *  Each layer's weights file were specified respectively.
 *  ******************************************************
 *
 *                      OR
 *
 *  ******************************************************
 *  All the layers' weights are in one file.
 *  We may locate the weights by a map, scuh as:
 *  weight name     : location, size
 *  conv_1          : 0,896
 *  block_1_expand  : 896,1728
 *  ******************************************************
 *
 */
- (instancetype)initWithPlist:(NSString *)plist;
- (instancetype)initWithDictionary:(NSDictionary *)dictionary;

/*
 *  Build up the whole model.
 *  1. Create and initialize each layer.
 *  2. Load the weights for the layers if there were.
 *  3. Connect the layers.
 */
- (void)compile:(id<MTLDevice>)device;

/*
 *  Loading the weights from files respectively.
 *  The file name should be identical to the weights name in the info plist.
 *
 *  This function may take a few seconds according to the weights size.
 */
- (void)loadWeights;

/*
 *  Loading the weights from one file.
 *  The map file for mapping the weights data range within the file.
 *
 *  This function may take a few seconds according to the weights size.
 */
- (void)loadWeights:(NSString *)weightsFile map:(NSDictionary *)weightsMap;
//  Loading map data from a json file.
- (void)loadWeights:(NSString *)weightsFile mapFile:(NSString *)mapFile;

/*
 *  Do the prediction, and callbacks.
 *  NOTE: The input texture will throw to the network directly,
 *        so it's your duty to make the preprocessing if needed.
 */
- (void)predict:(id<MTLTexture>)bgraU8Texture;

/*
 *  Model interacting function.
 */
- (MTLUInt2)inputSize;    // network input image size.
- (MetalTensorInputLayer *)inputLayer;
- (NSArray<MetalTensorOutputLayer *> *)outputLayers;
- (NSArray<MetalTensorLayer *> *)allLayers;
- (MetalTensorLayer *)layerWithName:(NSString *)name;
- (MetalTensorLayerDescriptor *)layerDescriptorWithName:(NSString *)name;

#ifdef DEBUG
// for debug
@property (nonatomic, assign) int verbose; // If you need the detail of running

- (void)setNeuronType:(NeuronType *)neuronType forLayerNamed:(NSString *)name; // call before compile
- (MetalTensorOutputLayer *)outputLayerWithName:(NSString *)layerName;
- (void)printTexture:(id<MTLTexture>)bgraU8Texture;

#endif

@end

NS_ASSUME_NONNULL_END
