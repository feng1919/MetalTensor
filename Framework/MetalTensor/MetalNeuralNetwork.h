//
//  MetalNeuralNetwork.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/25.
//  Copyright © 2019 fengshi. All rights reserved.
//

/*
 *  Tools for inference pre-trained neural networks.
 *
 */

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
 *  parallel computation pipeline, and takes a MetalImageTexture
 *  as an input, also may output to the targets.
 *
 *  One may sub class of this for customise need.
 *
 */

@interface MetalNeuralNetwork : MetalImageOutput <MetalImageInput> {
    
@protected
    // NETWORK
    NSDictionary *_networkDesc;
    
    // Typically, there is only one input tensor.
    MetalTensorInputLayer *_inputLayer;
    
    // All of the output layers
    NSMutableArray<MetalTensorOutputLayer *> *_outputLayers;
    
    // All of the layers, including the input layer and the output layers.
    // The key of the dictionary is layer name.
    NSMutableDictionary<NSString *, MetalTensorNode *> *_allLayers;
    
    // If the neural network were built up from a plist file,
    // the informations of plist are stored in this dictionary.
    // This may be empty, if it is not built up from plist.
    NSMutableDictionary<NSString *, MetalTensorLayerDescriptor *> *_allLayerDescriptors;
    
    // There may be several neural networks running in memory,
    // The reuse identifier is used to keep the individual neural network
    // memory-safety from the others.
    // Typically one just leave it alone.
    NSString *_reuseIdentifier;
    
    // sync between network and MetalImage
    dispatch_semaphore_t  _network_semaphore;
    dispatch_queue_t _network_queue;
    
    // input texture from MetalImage
    MetalImageTexture *_inputTexture;
}

/*
 *  Callbacks for the custom tasks.
 *
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

- (instancetype)init NS_UNAVAILABLE;

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
- (instancetype)initWithDictionary:(NSDictionary *)dictionary NS_DESIGNATED_INITIALIZER;

/*
 *  Build up the entire model.
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
 *  Make the prediction, you may add MetalTensorOutputLayer to output the tensor
 *  produced by the layer.
 *
 *  NOTE: The input texture will throw into the network directly,
 *        so one should preprocess the data if needed.
 *
 */
- (void)predict:(id<MTLTexture>)bgraU8Texture;

/*
 *  Network input image size.
 */
- (MTLUInt2)inputSize;
- (void)setInputSize:(MTLUInt2)size;

- (MetalTensorInputLayer *)inputLayer;
- (NSArray<MetalTensorOutputLayer *> *)outputLayers;
- (NSArray<MetalTensorNode *> *)allLayers;
- (MetalTensorNode *)layerWithName:(NSString *)name;
- (MetalTensorLayerDescriptor *)layerDescriptorWithName:(NSString *)name;
- (BOOL)removeLayerWithName:(NSString *)name;

- (MetalTensorOutputLayer *)outputLayerWithName:(NSString *)layerName;

#ifdef DEBUG
// for debug
@property (nonatomic, assign) int verbose; // If one need the detail of running

- (void)setNeuronType:(NeuronType *)neuronType forLayerNamed:(NSString *)name; // call before compile;
- (void)printTexture:(id<MTLTexture>)bgraU8Texture;

#endif

@end

NS_ASSUME_NONNULL_END
