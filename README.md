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

我们选择了 NGC 优化后的 TensorFlow、PyTorch、MxNet，作为性能的参考。

对这些框架的性能测试，我们选用相同的物理机执行，并严格参照各框架官网公布的测试方法进行复现。

### 1. [NGC TensorFlow 1.15](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags)

- 代码库：[DeepLearningExamples/TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow)

### 2. [NGC PyTorch](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch/tags)

- 代码库：[DeepLearningExamples/PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch)
### 3. [NGC MxNet](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet/tags)

- 代码库：[DeepLearningExamples/MxNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet)


## 三、测试结果

说明：

- 本次测试选用`8 * V100-SXM2-16GB`物理机做单机单卡、单机8卡测试，选用4台`8 * V100-SXM2-32GB`物理机做32卡测试。
- 测试中，我们尽可能复现不同框架的最好极限值，因此以下测试结果默认打来了各个框架的各种加速功能/选项，如：
   - 对于支持 `DALI/XLA` 的框架，以下测试为开启 `DALI/XLA` 的数据

### 1. ResNet50V1.5
> 详细数据请见[《Paddle ResNet50V1.5 性能测试报告》](./ResNet50V1.5)

- **单位**：`images/sec`

- FP32测试

| 参数 | PaddlePaddle | NGC TensorFlow 1.15 | NGC PyTorch | NGC MXNet |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| FP32 GPU=1,BS=128 | 383(BS=96) | 408.3 | 364.2 | 387.1(BS=96) |
| FP32 GPU=8,BS=128 | 2753.3(BS=96) | 3105.3 | 2826.8 | 2998.1(BS=96) |
| FP32 GPU=32,BS=128 | -(BS=96) | - | - | -(BS=96) |

- AMP测试

| 参数 | PaddlePaddle | NGC TensorFlow 1.15 | NGC PyTorch | NGC MXNet |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| AMP GPU=1,BS=128 | 1335.1 | 954.9 | 828.7 | 1380.6 |
| AMP GPU=1,BS=256 | 1400.1(BS=208) | 978.8 | 841.6 | 1447.6(BS=192) |
| AMP GPU=8,BS=128 | 8322.9 | 7829.9	 | 6014.7 | 9218.9 |
| AMP GPU=8,BS=256 | 9099.5(BS=208) | 7810.8 | 6230.1(BS=248) | 9765.6(BS=192) |
| AMP GPU=32,BS=128 | 27500.8 | - | - | - |
| AMP GPU=32,BS=256 | -(BS=208) | - | - | -(BS=192) |

> 以上测试，由于显存限制，下调了部分测试的BatchSize，并在表格中注明 <br>
> Pytorch AMP 8卡在BatchSize=256时会OOM，因此下调BatchSize为248

### 2. Bert Base Pre-Training
> 详细数据请见[《Paddle Bert Base 性能测试报告》](./Bert)

- **max_seq_len**: `128`
- **单位**：`sequences/sec`

- FP32测试

| 参数 | [PaddlePaddle](./Bert) | [NGC TensorFlow 1.15](./Bert/OtherReports/TensorFlow) | [NGC PyTorch](./Bert/OtherReports/PyTorch) |
|:-----:|:-----:|:-----:|:-----:|
| FP32 GPU=1,BS=32 | 147.14 | 142.67 |  |
| FP32 GPU=1,BS=48 | 153.47 | 148.23 | 128.92 |
| FP32 GPU=8,BS=32 |1140.52 |  |  |
| FP32 GPU=8,BS=48 |1186.89  |  |  |
| FP32 GPU=32,BS=32 | 4541.76 | - | - |
| FP32 GPU=32,BS=48 | 5026.88 | - | - |

- AMP测试

| 参数 | [PaddlePaddle](./Bert) | [NGC TensorFlow 1.15](./Bert/OtherReports/TensorFlow) | [NGC PyTorch](./Bert/OtherReports/PyTorch) |
|:-----:|:-----:|:-----:|:-----:|
| AMP GPU=1,BS=64 | 595.49 | 488.32 | 524.48 |
| AMP GPU=1,BS=96 | 628.25 | 536.06 | 543.76 |
| AMP GPU=8,BS=64 | 4329.79 |  | 4058.34|
| AMP GPU=8,BS=96 | 4569.42 |  | 4208.12|
| AMP GPU=32,BS=64 | 17756.8 | - | - |
| AMP GPU=32,BS=96 | 19171.84 | - | - |
