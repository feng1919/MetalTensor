# MetalTensor

[English](README.md) | [简体中文](README.zh-CN.md)

MetalTensor is a lightweight neural network inference framework built on Apple Metal / Metal Performance Shaders for real-time vision workloads on iOS. It uses `plist` files to describe network structure, supports loading weights either from separate files or a single binary blob, and plugs directly into the `MetalImage` video processing pipeline.

The repository currently includes three complete demos:

1. **MobileNetV2**: real-time image classification
2. **PortraitSegment**: portrait segmentation
3. **RapidFaceDetect**: face detection with 68 facial landmarks

> **Note**
>
> 1. `MetalTensor` is now distributed through Swift Package Manager and resolves `MetalImage` from `https://github.com/feng1919/MetalImage`
> 2. The legacy demo projects in `Examples/` are still Xcode-project based
> 3. Metal code must run on a real device; **the simulator is not supported**

## Use Cases

MetalTensor is a good fit for:

- deploying trained CNN models to iPhone / iPad
- processing live camera frames with a GPU preprocessing + inference + rendering pipeline
- building inference graphs from configuration files instead of wiring every layer manually
- reusing the same inference infrastructure across classification, segmentation, and detection tasks

## Project Layout

```text
.
├── Framework/              # MetalTensor framework sources and Xcode project
│   └── MetalTensor/
├── Package.swift           # Swift Package Manager manifest
├── Examples/
│   ├── MobileNetV2/        # Image classification demo
│   ├── PortraitSegment/    # Portrait segmentation demo
│   └── RapidFaceDetect/    # Face detection / landmarks demo
└── LICENSE
```

## Core Features

- inference runtime centered on `MetalNeuralNetwork`
- network topology in `plist`, weight-range mapping in `json`, and weight blobs in `bin`
- synchronous and asynchronous inference
- `float16` / `float32` precision selection
- direct input from either `MTLTexture` or `MetalTensor`
- output retrieval through output layers or completion callbacks
- built-in SSD decoding utilities for detection workflows

## Swift Package Manager

Add `MetalTensor` as a package dependency:

```swift
.package(url: "https://github.com/feng1919/MetalTensor", branch: "master")
```

Then depend on the `MetalTensor` product:

```swift
.product(name: "MetalTensor", package: "MetalTensor")
```

`MetalImage` is pulled in automatically by `MetalTensor` through Swift Package Manager, so you no longer need to vendor `MetalImage.framework` inside your app or framework target.

Until release tags are published, use the `master` branch in Swift Package Manager.

Layer types visible in the current codebase include:

- `input`
- `output`
- `convolution`
- `dense`
- `softmax`
- `pooling_average`
- `pooling_max`
- `reshape`
- `concatenate`
- `inverted_residual`
- `arithmetic`
- `neuron`
- `trans_conv`

## Running the Demos

### 1. Open a project

You can open any of these demo projects directly:

- `Examples/MobileNetV2/MobileNetV2.xcodeproj`
- `Examples/PortraitSegment/PortraitSegment.xcodeproj`
- `Examples/RapidFaceDetect/RapidFaceDetect.xcodeproj`

Each demo references the local `Framework/MetalTensor.xcodeproj`. The Swift package target uses the remote `MetalImage` package dependency instead of the vendored binary framework.

### 2. Select a physical device

The demos depend on Metal and camera input, so they need to run on an iPhone / iPad device.

### 3. Grant permissions on first launch

Camera-based demos require camera permission the first time they are launched.

## Integrating a Custom Model

### 1. Prepare model assets

A typical setup uses three files:

- `YourModel.plist`: network structure description
- `YourModel.bin`: merged weights file
- `YourModel.json`: mapping from weight names to binary ranges

If your weights are stored per layer, you can also use `loadWeights` to load them individually.

### 2. Define a network subclass

The common pattern is to subclass `MetalNeuralNetwork`:

```objective-c
#import <MetalTensor/MetalNeuralNetwork.h>

@interface YourNet : MetalNeuralNetwork
@end

@implementation YourNet

- (instancetype)init {
    NSString *plist = [[NSBundle mainBundle] pathForResource:@"YourModel" ofType:@"plist"];
    return [self initWithPlist:plist];
}

- (void)loadWeights {
    NSString *weights = [[NSBundle mainBundle] pathForResource:@"YourModel" ofType:@"bin"];
    NSString *map = [[NSBundle mainBundle] pathForResource:@"YourModel" ofType:@"json"];
    [self loadWeights:weights mapFile:map];
}

@end
```

### 3. Compile the network and load weights

```objective-c
YourNet *net = [[YourNet alloc] init];
net.synchronizedProcessing = NO;   // optional
net.dataType = MPSDataTypeFloat16; // float16 is the default

[net compile:[MetalDevice sharedMTLDevice]];
[net loadWeights];
```

### 4. Run inference

If you are already inside a `MetalImage` pipeline, the network can be added as a node. If you already have a texture, you can call:

```objective-c
[net predict:bgraTexture];
```

Tensor input is also supported:

```objective-c
[net predictWithTensor:tensor];
```

### 5. Read the output

Two common patterns are:

1. attach a `completedHandler` and read outputs when the command buffer finishes
2. fetch a `MetalTensorOutputLayer` and continue processing its `MPSImage` / tensor output

`Examples/MobileNetV2/MobileNetV2/MobileNetV2.m` shows how to convert `float16` output into classification results inside `completedHandler`.

## `plist` Network Description Example

`MobileNetV2_1.0.plist` shows the basic MetalTensor graph format:

```xml
<key>input</key>
<dict>
    <key>inputs</key>
    <string>224, 224, 3</string>
    <key>targets</key>
    <string>preprocess</string>
    <key>type</key>
    <string>input</string>
</dict>
```

Each node typically defines:

- `type`: layer type
- `inputs`: input shape
- `targets`: downstream layers
- `weight` / `weights`: weight names
- `kernel` / `stride` / `padding` / `activation`: layer parameters

During `compile:`, the framework will:

1. parse the configuration
2. instantiate layer descriptors
3. create the corresponding layer objects
4. connect the full computation graph

## Demo Overview

### MobileNetV2

- captures camera frames
- crops the center region and resizes it to the network input size
- overlays classification results and FPS on screen

Key files:

- `Examples/MobileNetV2/MobileNetV2/MobileNetV2.m`
- `Examples/MobileNetV2/MobileNetV2/ViewController.m`

### PortraitSegment

- sends camera frames into a segmentation network
- outputs a single-channel portrait mask
- composites the mask with the original image in a custom fragment shader

Key files:

- `Examples/PortraitSegment/PortraitSegment/PortraitSegmentNet.h`
- `Examples/PortraitSegment/PortraitSegment/PortraitSegmentFilter.m`
- `Examples/PortraitSegment/PortraitSegment/ViewController.m`

### RapidFaceDetect

- runs face detection first
- predicts facial landmarks on the detected face region
- draws bounding boxes and landmark points back onto the image

Key files:

- `Examples/RapidFaceDetect/RapidFaceDetect/RapidFaceDetectNet.h`
- `Examples/RapidFaceDetect/RapidFaceDetect/RapidFaceDetectFilter.m`
- `Examples/RapidFaceDetect/RapidFaceDetect/LandmarksNet.h`
- `Examples/RapidFaceDetect/RapidFaceDetect/LandmarksFilter.m`

## Relationship with MetalImage

MetalTensor is designed around the `MetalImage` ecosystem. `MetalNeuralNetwork` subclasses `MetalImageOutput` and conforms to `MetalImageInput`, so it can be inserted into an existing GPU image pipeline like a regular `MetalImage` filter.

That is why the demo projects use chain-style connections like this:

```objective-c
[videoCamera addTarget:cropFilter];
[cropFilter addTarget:resizeFilter];
[resizeFilter addTarget:net];
```

If your app already uses `MetalImage`, integration is relatively straightforward.

## Known Limitations

- the current repository is primarily organized around iOS framework code and demos
- demo projects depend on the bundled `MetalImage.framework`
- there is no model conversion tool in this repository; model descriptors and weights must be prepared separately
- this repository focuses on **inference**, not training

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE).
