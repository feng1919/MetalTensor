//
//  MetalNeuralNetwork.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/25.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MetalNeuralNetwork.h"
#import "NSString+Extension.h"
#import "MIConvolutionLayer.h"
#import "MetalTensorInputLayer.h"
#import "MetalTensorOutputLayer.h"
#import "MIReshapeLayer.h"
#import "MIConcatenateLayer.h"
#import "MITemporaryImageCache.h"
#import <MetalImage/MetalDevice.h>

@interface MetalNeuralNetwork()

@end

@implementation MetalNeuralNetwork

- (instancetype)init {
    NSAssert(NO, @"Invalid initialize function, use -initWithPlist: or -initWithDictionary:");
    return nil;
}

- (instancetype)initWithPlist:(NSString *)infoPlist {
    return [self initWithDictionary:[NSDictionary dictionaryWithContentsOfFile:infoPlist removingBlankSpaces:YES]];
}

- (instancetype)initWithDictionary:(NSDictionary *)dictionary {
    if (self = [super init]) {
        _networkDesc = dictionary;
        _synchronizedProcessing = YES;
        _scheduledHandler = NULL;
        _completedHandler = NULL;
        _network_queue = dispatch_queue_create("metal_neuron_network_queue", DISPATCH_QUEUE_SERIAL);
        _network_semaphore = dispatch_semaphore_create(1);
        NSInteger identifier = [[MITemporaryImageCache sharedCache] registerReuseIdentifier];
        _reuseIdentifier = [NSString stringWithFormat:@"%ld", identifier];
    }
    return self;
}

- (void)dealloc {
    [[MITemporaryImageCache sharedCache] unregisterReuseIdentifier:_reuseIdentifier.integerValue];
}

- (void)compile:(id<MTLDevice>)device {
    
    _allLayers = [NSMutableDictionary dictionary];
    _allLayerDescriptors = [NSMutableDictionary dictionary];
    
    // generating and compiling all of the layers...
    for (NSString *key in _networkDesc.allKeys) {
        id entity = _networkDesc[key];
        if (![entity isKindOfClass:[NSDictionary class]]) {
            continue;
        }
        
        NSString *type = entity[@"type"];
        Class descriptorClass = DescriptorWithType(type);
        MetalTensorLayerDescriptor *desc = [[descriptorClass alloc] initWithDictionary:entity];
        desc.name = key;
        [_allLayerDescriptors setObject:desc forKey:key];
        
        Class layerClass = LayerWithType(desc.type);
        MetalTensorNode *layer = [[layerClass alloc] initWithDescriptor:desc];
        NSParameterAssert(layer);
        [layer compile:device];
        [layer setLabel:key];
        [_allLayers setObject:layer forKey:key];
    }
    
    // connecting the layers...
    for (NSString *layerName in _allLayerDescriptors.allKeys) {
        MetalTensorNode *layer = _allLayers[layerName];
        MetalTensorLayerDescriptor *desc = _allLayerDescriptors[layerName];
        NSParameterAssert(desc.targetIndices == nil || desc.targetIndices.count == desc.targets.count);
        for (int i = 0; i < desc.targets.count; i++) {
            NSString *target = desc.targets[i];
            MetalTensorNode *targetLayer = _allLayers[target];
            if (targetLayer) {
                if ([targetLayer conformsToProtocol:@protocol(MetalTensorInput)]) {
                    // If the index were specified
                    if (desc.targetIndices.count == desc.targets.count) {
                        [layer addTarget:(id<MetalTensorInput>)targetLayer atTempImageIndex:[desc.targetIndices[i] intValue]];
                    }
                    else {
                        [layer addTarget:(id<MetalTensorInput>)targetLayer];
                    }
                }
                else {
                    NSAssert(NO, @"The layer %@ does not conform to the protocol <MetalTensorInput>", target);
                }
            }
            else {
                NSAssert(NO, @"Missing target layer: %@", target);
            }
        }
    }
    
    // Obtain the input layer.
    for (NSString *layerName in _allLayers) {
        MetalTensorLayerDescriptor *descriptor = _allLayerDescriptors[layerName];
        NSAssert(descriptor, @"Failed to obtain descriptor for layer: %@", layerName);
        if ([descriptor.type isEqualToString:@"input"]) {
            _inputLayer = (MetalTensorInputLayer *)_allLayers[layerName];
            break;
        }
    }
    NSAssert(_inputLayer, @"There must be one input layer.");
    
    // Obtain the output layers.
    _outputLayers = [NSMutableArray array];
    for (NSString *layerName in _allLayers) {
        MetalTensorLayerDescriptor *descriptor = _allLayerDescriptors[layerName];
        NSAssert(descriptor, @"Failed to obtain descriptor for layer: %@", layerName);
        if ([descriptor.type isEqualToString:@"output"]) {
            [_outputLayers addObject:(MetalTensorOutputLayer *)_allLayers[layerName]];
        }
    }
    NSAssert(_outputLayers.count > 0, @"There must be at least one output layer.");
    
#ifdef DEBUG
    // Setting the console log verbose for all of the layers.
    self.verbose = _verbose;
#endif
    
    printf("\n\n\n");
    printf("\n==================== MetalNeuralNetwork ====================");
    printf("\n%s [Total layers: %d]", NSStringFromClass([self class]).UTF8String, (int)_allLayers.allValues.count);
    printf("\n============================================================");
    printf("\n\n\n");
}

- (void)loadWeights {
    for (MetalTensorLayer *layer in _allLayers) {
        if ([layer conformsToProtocol:@protocol(MetalTensorWeights)]) {
            [(id<MetalTensorWeights>)layer loadWeights];
        }
    }
}

- (void)loadWeights:(NSString *)weightsFile mapFile:(NSString *)mapFile {
    
    NSParameterAssert([[NSFileManager defaultManager] fileExistsAtPath:mapFile]);
    
    NSData *mapData = [NSData dataWithContentsOfFile:mapFile];
    NSDictionary *map = [NSJSONSerialization JSONObjectWithData:mapData options:0 error:nil];
    [self loadWeights:weightsFile map:[map removeBlankSpaces]];
}

- (void)loadWeights:(NSString *)weightsFile map:(NSDictionary *)weightsMap {
    
    NSParameterAssert([[NSFileManager defaultManager] fileExistsAtPath:weightsFile]);
    
    for (NSString *layerName in _allLayerDescriptors.allKeys) {
        
        if (![_allLayers[layerName] conformsToProtocol:@protocol(MetalTensorWeights)]) {
            continue;
        }
        
        id<MetalTensorWeights> layer = (id<MetalTensorWeights>)_allLayers[layerName];
        if ([layer didLoadWeights]) {
            continue;
        }
        
        MetalTensorLayerDescriptor *desc = _allLayerDescriptors[layerName];
        if ([desc.type isEqualToString:@"convolution"]) {
            MIConvolutionLayerDescriptor *convDesc = (MIConvolutionLayerDescriptor *)desc;
            NSAssert(weightsMap[convDesc.weight], @"%@ weights range not found. Check the json file.", convDesc.name);
            NSArray<NSString *> *rangeComponents = [weightsMap[convDesc.weight] componentsSeparatedByString:@","];
            NSUInteger location = [rangeComponents[0] integerValue];
            NSUInteger length = [rangeComponents[1] integerValue];
            NSRange range = NSMakeRange(location, length);
            
            [layer loadWeights:weightsFile range:&range];
        }
        else if ([desc.type isEqualToString:@"dense"]) {
            MIFullyConnectedLayerDescriptor *denseDesc = (MIFullyConnectedLayerDescriptor *)desc;
            NSAssert(weightsMap[denseDesc.weight], @"%@ weights range not found. Check the json file.", denseDesc.name);
            NSArray<NSString *> *rangeComponents = [weightsMap[denseDesc.weight] componentsSeparatedByString:@","];
            NSUInteger location = [rangeComponents[0] integerValue];
            NSUInteger length = [rangeComponents[1] integerValue];
            NSRange range = NSMakeRange(location, length);
            
            [layer loadWeights:weightsFile range:&range];
        }
        else if ([desc.type isEqualToString:@"inverted_residual"]) {
            MIInvertedResidualModuleDescriptor *irmDesc = (MIInvertedResidualModuleDescriptor *)desc;
            NSRange ranges[3];
            for (int i = 0; i < 3; i++) {
                NSAssert(weightsMap[irmDesc.weights[i]], @"%@ weights range not found. Check the json file.", irmDesc.name);
                NSArray<NSString *> *rangeComponents = [weightsMap[irmDesc.weights[i]] componentsSeparatedByString:@","];
                ranges[i].location = [rangeComponents[0] integerValue];
                ranges[i].length = [rangeComponents[1] integerValue];
            }
            
            [layer loadWeightsList:@[weightsFile, weightsFile, weightsFile] rangeList:ranges];
        }
        else if ([desc.type isEqualToString:@"trans_conv"]) {
            MITransposeConvolutionLayerDescriptor *convDesc = (MITransposeConvolutionLayerDescriptor *)desc;
            NSAssert(weightsMap[convDesc.weight], @"%@ weights range not found. Check the json file.", convDesc.name);
            NSArray<NSString *> *rangeComponents = [weightsMap[convDesc.weight] componentsSeparatedByString:@","];
            NSUInteger location = [rangeComponents[0] integerValue];
            NSUInteger length = [rangeComponents[1] integerValue];
            NSRange range = NSMakeRange(location, length);
            
            [layer loadWeights:weightsFile range:&range];
        }
    }
}

- (void)predict:(id<MTLTexture>)bgraU8Texture
{
    [_inputLayer inputTexture:bgraU8Texture];
    
    id<MTLCommandQueue> command_queue = [MetalDevice sharedCommandQueue];
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    [command_buffer setLabel:_reuseIdentifier];
//    [command_buffer enqueue];
    [[MITemporaryImageCache sharedCache] beginContextWithCommandBuffer:command_buffer];
    [_inputLayer processOnCommandBuffer:command_buffer];
    [[MITemporaryImageCache sharedCache] endContextWithCommandBuffer:command_buffer];
    
    if (_scheduledHandler) {
        [command_buffer addScheduledHandler:_scheduledHandler];
    }
    if (_completedHandler) {
        [command_buffer addCompletedHandler:_completedHandler];
    }
    [command_buffer commit];
}

- (MTLUInt2)inputSize {
    DataShape *dataShape = [_inputLayer outputShapeRef];
    return MTLUInt2Make(dataShape->row, dataShape->column);
}

- (MetalTensorInputLayer *)inputLayer {
    return _inputLayer;
}

- (NSArray<MetalTensorOutputLayer *> *)outputLayers {
    return _outputLayers;
}

- (NSArray<MetalTensorNode *> *)allLayers {
    return _allLayers.allValues;
}

- (MetalTensorNode *)layerWithName:(NSString *)name {
    return _allLayers[name];
}

- (MetalTensorLayerDescriptor *)layerDescriptorWithName:(NSString *)name {
    return _allLayerDescriptors[name];
}

- (BOOL)removeLayerWithName:(NSString *)name {
    MetalTensorNode *layer = _allLayers[name];
    if (!layer) {
        return NO;
    }
    
    if (![layer conformsToProtocol:@protocol(MetalTensorInput)]) {
        // It is not necessary to remove from.
        return NO;
    }
    
    for (MetalTensorNode *node in _allLayers) {
        [node removeTarget:(MetalTensorNode <MetalTensorInput>*)layer];
    }
    
    return YES;
}

- (MetalTensorOutputLayer *)outputLayerWithName:(NSString *)layerName {
    NSParameterAssert([layerName length] > 0);
    MetalTensorNode *layer = [self layerWithName:layerName];
    NSParameterAssert(layer);
    DataShape output_shape = [layer outputShape];
    MetalTensorOutputLayer *outputLayer = [[MetalTensorOutputLayer alloc] initWithInputShape:&output_shape];
    [layer addTarget:outputLayer];
    return outputLayer;
}

#ifdef DEBUG
- (void)setNeuronType:(NeuronType *)neuronType forLayerNamed:(NSString *)name {
    if (_networkDesc[name] == nil) {
        NSLog(@"Layer name %@ not found.", name);
        return;
    }
    
    NSMutableDictionary *layerDict = [NSMutableDictionary dictionaryWithDictionary:_networkDesc[name]];
    layerDict[@"activation"] = [NSString stringWithFormat:@"%d,%0.2f,%0.2f", neuronType->neuron, neuronType->a, neuronType->b];
    
    NSMutableDictionary *newNetworkDesc = [NSMutableDictionary dictionaryWithDictionary:_networkDesc];
    newNetworkDesc[name] = layerDict;
    _networkDesc = [NSDictionary dictionaryWithDictionary:newNetworkDesc];
}

#endif

#pragma mark - MetalImageInput delegate

- (void)setInputTexture:(MetalImageTexture *)newInputFramebuffer atIndex:(NSInteger)textureIndex {
    _inputTexture = newInputFramebuffer;
    [_inputTexture lock];
}

- (void)setInputRotation:(MetalImageRotationMode)newInputRotation atIndex:(NSInteger)textureIndex {
}

- (void)newTextureReadyAtTime:(CMTime)frameTime atIndex:(NSInteger)textureIndex {
    
    if (!_synchronizedProcessing) {
        if (dispatch_semaphore_wait(_network_semaphore, DISPATCH_TIME_NOW) != 0) {
            NSLog(@"Drop off one frame");
            [_inputTexture unlock];
            return;
        }
    }
    
    [MetalDevice commitCommandBuffer];
    
    if (_synchronizedProcessing) {
        [self predict:_inputTexture.texture];
        [_inputTexture unlock];
    }
    else {
        __weak __auto_type weakSelf = self;
        dispatch_async(_network_queue, ^{
            __strong __auto_type strongSelf = weakSelf;
            [strongSelf predict:strongSelf->_inputTexture.texture];
            [strongSelf->_inputTexture unlock];
            
            dispatch_semaphore_signal(strongSelf->_network_semaphore);
        });
    }
}

#ifdef DEBUG

- (void)printTexture:(id<MTLTexture>)texture {

    int width = (int)texture.width;
    int height = (int)texture.height;
    size_t imageByteCount = width * height * 4 * sizeof(uint8_t);
    uint8_t *result = (uint8_t *)malloc(imageByteCount);
    NSUInteger bytesPerRow = width * 4 * sizeof(uint8_t);
    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    [texture getBytes:result bytesPerRow:bytesPerRow fromRegion:region mipmapLevel:0];
    
    printf("\n");
    for (int i = 0; i < height; i++) {
        printf("\nrow[%d]:\n", i);
        int row = i * width * 4;
        for (int j = 0; j < width; j++) {
            int index = row + j * 4;
            printf("(%d, %d, %d) ", result[index], result[index+1], result[index+2]);
        }
    }
    printf("\n");
    
    free(result);
}

- (void)setVerbose:(int)verbose {
    _verbose = verbose;
    
    for (MetalTensorLayer *layer in _allLayers.allValues) {
        layer.verbose = verbose;
    }
}

#endif

@end
