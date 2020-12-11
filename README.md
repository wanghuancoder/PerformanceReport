<!-- omit in toc -->
# Paddle Performance Report——Paddle框架性能测试报告


本 repo 用于公开 PaddlePaddle 开源实现的各个学术界、工业界前沿模型，在训练期间的性能数据，同时提供了各模型性能测试的详细复现流程，以供参考。

同时，我们也在相同的硬件执行环境下，按照业内其它知名深度学习框架公开的代码和教程，测试了对应模型的性能数据，并记录具体日志和数据。

<!-- omit in toc -->
## 目录

- [一、测试模型](#一测试模型)
  - [1.计算机视觉](#1计算机视觉)
  - [2.自然语言处理](#2自然语言处理)
- [二、供对比的业内深度学习框架](#二供对比的业内深度学习框架)
  - [1. NGC TensorFlow 1.15](#1-ngc-tensorflow-115)
  - [2. NGC PyTorch](#2-ngc-pytorch)
  - [3. NGC MxNet](#3-ngc-mxnet)
  - [4. OneFlow](#4-oneflow)
- [三、测试结果](#三测试结果)
  - [1. ResNet50V1.5](#1-resnet50v15)
  - [2. Bert Base Pre-Training](#2-bert-base-pre-training)

## 一、测试模型

目前我们公开了**计算机视觉**和**自然语言处理**领域的两个典型模型的性能对比数据：

### 1.计算机视觉
- [ResNet50V1.5](./ResNet50V1.5)

### 2.自然语言处理
- [Bert Base Pre-Training](./Bert)

我们将持续开展性能测试工作，后续将逐步公开更多性能数据，敬请期待。

## 二、供对比的业内深度学习框架

我们选择了 NGC 优化后的 TensorFlow、PyTorch、MxNet，以及国内优秀的深度学习框架 OneFlow 等代码实现，作为性能的参考。

对这些框架的性能测试，我们选用相同的物理机执行，并严格参照各框架官网公布的测试方法进行复现。

### 1. [NGC TensorFlow 1.15](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags)

- 代码库：[DeepLearningExamples/TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow)

### 2. [NGC PyTorch](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch/tags)

- 代码库：[DeepLearningExamples/PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch)
### 3. [NGC MxNet](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet/tags)

- 代码库：[DeepLearningExamples/MxNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet)

### 4. [OneFlow](https://github.com/Oneflow-Inc/oneflow/tree/v0.2.0)

- 代码库：[Oneflow-Inc/DLPerf/OneFlow/](https://github.com/Oneflow-Inc/DLPerf/tree/master/OneFlow)


## 三、测试结果

说明：

- GPU型号：`V100-SXM2-32GB`
- 测试中，我们尽可能复现不同框架的最好极限值，因此以下测试结果默认打来了各个框架的各种加速功能/选项，如：
   - 对于支持 `DALI/XLA` 的框架，以下测试为开启 `DALI/XLA` 的数据

### 1. ResNet50V1.5

> 详细数据请见[《Paddle ResNet50V1.5 性能测试报告》](./ResNet50V1.5)

- 单位：`images/sec`

| 参数 | PaddlePaddle | NGC TensorFlow 1.15 | NGC PyTorch | NGC MXNet |
|-----|-----|-----|-----|-----|
| FP32 GPU=1,BS=128 | 337.5 | 398.349 | 356.77 | 381.46 |
| FP32 GPU=1,BS=256 | 337.5 | 414.631 | 372.76 | 384.72 |
| AMP GPU=1,BS=128 | - | 979.091 | 782.43 | 1316.7 |
| AMP GPU=1,BS=256 | - | 994.131 | 798.15 | 1412 |
| FP32 GPU=8,BS=128 | - | 3038.730 | 2742.41 | 2948.2 |
| FP32 GPU=8,BS=256 | - | 3210.793 | 2884.68 | 2987.9 |
| AMP GPU=8,BS=128 | - | 7601.888 | 5715.58 | 9500.5 |
| AMP GPU=8,BS=256 | - | 7799.018 | 6007.61 | 10440 |
| FP32 GPU=32,BS=128 | - | - | - | - |
| FP32 GPU=32,BS=256 | - | - | - | - |
| AMP GPU=32,BS=128 | - | - | - | - |
| AMP GPU=32,BS=256 | - | - | - | - |

### 2. Bert Base Pre-Training
> 详细数据请见[《Paddle Bert Base 性能测试报告》](./Bert)


- 单位：`sequences/sec`


| 参数 | PaddlePaddle | NGC TensorFlow 1.15 | NGC PyTorch |
|-----|-----|-----|-----|
| FP32 GPU=1,BS=64 | - | 153.94 | 127.02 |
| AMP GPU=1,BS=128 | - | 537.82 | 529.46 |
| FP32 GPU=8,BS=64 | - | - | - |
| AMP GPU=8,BS=128 | - | - | - |
| FP32 GPU=32,BS=64 | - | - | - |
| AMP GPU=32,BS=128 | - | - | - |
