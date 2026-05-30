# MetalTensor

[English](README.md) | [简体中文](README.zh-CN.md)

MetalTensor 是一个基于 Apple Metal / Metal Performance Shaders 的轻量级神经网络推理框架，面向 iOS 端实时视觉任务。它用 `plist` 描述网络结构，支持把模型权重从独立文件或单个二进制文件加载进来，并且可以直接接入 `MetalImage` 的视频处理管线。

仓库内同时提供了 3 个完整示例：

1. **MobileNetV2**：实时图像分类
2. **PortraitSegment**：人像分割
3. **RapidFaceDetect**：人脸检测与 68 点关键点

> **注意**
>
> 1. 当前工程依赖 `ThirdParts/MetalImage.framework`
> 2. 当前仓库未提供 CocoaPods / Swift Package Manager / Carthage 分发配置
> 3. Metal 相关代码需要在真机运行，**不支持 Simulator**

## 适用场景

MetalTensor 适合这类工作：

- 把已经训练好的 CNN 模型部署到 iPhone / iPad
- 从摄像头实时取帧，走 GPU 预处理 + 推理 + 渲染链路
- 用配置文件快速搭建推理图，而不是为每个模型手写整套层连接代码
- 在分类、分割、检测等任务里复用统一的推理基础设施

## 项目结构

```text
.
├── Framework/              # MetalTensor 框架源码与 Xcode 工程
│   └── MetalTensor/
├── Examples/
│   ├── MobileNetV2/        # 图像分类示例
│   ├── PortraitSegment/    # 人像分割示例
│   └── RapidFaceDetect/    # 人脸检测 / 关键点示例
├── ThirdParts/
│   └── MetalImage.framework
└── LICENSE
```

## 核心能力

- 基于 `MetalNeuralNetwork` 的网络编译与执行框架
- 用 `plist` 描述网络拓扑，用 `json` 映射权重范围，用 `bin` 存放权重数据
- 支持同步 / 异步推理
- 支持 `float16` / `float32` 精度切换
- 可直接接收 `MTLTexture` 或 `MetalTensor` 作为输入
- 可从输出层取回结果，或通过回调把结果接回渲染链路
- 内置 SSD 解码相关工具，便于做目标检测

当前代码中可见的网络层类型包括：

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

## 快速运行示例

### 1. 打开工程

可直接打开任一示例工程：

- `Examples/MobileNetV2/MobileNetV2.xcodeproj`
- `Examples/PortraitSegment/PortraitSegment.xcodeproj`
- `Examples/RapidFaceDetect/RapidFaceDetect.xcodeproj`

示例工程会引用：

- `Framework/MetalTensor.xcodeproj`
- `ThirdParts/MetalImage.framework`

### 2. 选择真机

示例依赖摄像头和 Metal，需选择 iPhone / iPad 真机运行。

### 3. 首次运行权限

摄像头相关示例第一次启动时需要授予相机权限。

## 自定义模型的接入方式

### 1. 准备模型资源

典型情况下需要 3 类文件：

- `YourModel.plist`：网络结构描述
- `YourModel.bin`：合并后的权重文件
- `YourModel.json`：权重名到二进制区间的映射

如果每层权重独立存放，也可以用 `loadWeights` 逐层加载。

### 2. 定义网络子类

最常见的做法是继承 `MetalNeuralNetwork`：

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

### 3. 编译并加载权重

```objective-c
YourNet *net = [[YourNet alloc] init];
net.synchronizedProcessing = NO;   // 可选
net.dataType = MPSDataTypeFloat16; // 默认即为 float16

[net compile:[MetalDevice sharedMTLDevice]];
[net loadWeights];
```

### 4. 执行推理

如果你已经在 `MetalImage` 管线中，可以直接把网络当作一个节点接进去；如果手里是纹理，也可以直接调用：

```objective-c
[net predict:bgraTexture];
```

也可以输入张量：

```objective-c
[net predictWithTensor:tensor];
```

### 5. 读取结果

常见方式有两种：

1. 给 `completedHandler` 挂回调，在命令缓冲完成后取输出
2. 获取某个 `MetalTensorOutputLayer`，把 `MPSImage` / tensor 拿出来继续处理

`Examples/MobileNetV2/MobileNetV2/MobileNetV2.m` 演示了在 `completedHandler` 中把 `float16` 输出转成分类结果的做法。

## `plist` 网络描述示例

`MobileNetV2_1.0.plist` 展示了 MetalTensor 的基本描述方式：

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

每个层节点通常会定义：

- `type`：层类型
- `inputs`：输入形状
- `targets`：输出连接到哪些层
- `weight` / `weights`：权重名称
- `kernel` / `stride` / `padding` / `activation`：层参数

框架会在 `compile:` 阶段完成：

1. 解析配置
2. 实例化层描述器
3. 创建对应层对象
4. 连接整张计算图

## 示例说明

### MobileNetV2

- 从摄像头取帧
- 中间区域裁剪后缩放到网络输入尺寸
- 将分类结果和 FPS 叠加到界面上

关键文件：

- `Examples/MobileNetV2/MobileNetV2/MobileNetV2.m`
- `Examples/MobileNetV2/MobileNetV2/ViewController.m`

### PortraitSegment

- 把摄像头帧送入分割网络
- 输出单通道人像 mask
- 在自定义 fragment shader 中把 mask 和原图合成

关键文件：

- `Examples/PortraitSegment/PortraitSegment/PortraitSegmentNet.h`
- `Examples/PortraitSegment/PortraitSegment/PortraitSegmentFilter.m`
- `Examples/PortraitSegment/PortraitSegment/ViewController.m`

### RapidFaceDetect

- 先做人脸检测
- 再对检测框区域做人脸关键点预测
- 最终将框和关键点绘制回图像

关键文件：

- `Examples/RapidFaceDetect/RapidFaceDetect/RapidFaceDetectNet.h`
- `Examples/RapidFaceDetect/RapidFaceDetect/RapidFaceDetectFilter.m`
- `Examples/RapidFaceDetect/RapidFaceDetect/LandmarksNet.h`
- `Examples/RapidFaceDetect/RapidFaceDetect/LandmarksFilter.m`

## 与 MetalImage 的关系

MetalTensor 本身是为 `MetalImage` 生态设计的。`MetalNeuralNetwork` 继承自 `MetalImageOutput`，并实现了 `MetalImageInput`，因此可以像普通 `MetalImage` filter 一样被插入已有 GPU 图像管线。

这也是示例工程里常见这种链式连接方式的原因：

```objective-c
[videoCamera addTarget:cropFilter];
[cropFilter addTarget:resizeFilter];
[resizeFilter addTarget:net];
```

如果你的应用已经在使用 `MetalImage`，接入成本会比较低。

## 已知限制

- 当前仓库主要以 iOS 工程和示例为主
- 示例工程依赖仓库内自带的 `MetalImage.framework`
- 未提供模型转换工具；模型描述文件和权重需要你自行准备
- README 中未覆盖训练流程，仓库重点是 **推理**

## License

本项目基于 **MIT License** 发布，详见 [LICENSE](LICENSE)。
