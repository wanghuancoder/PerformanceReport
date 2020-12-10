<!-- omit in toc -->
# Paddle ResNet50V1.5 性能测试

此处给出了Paddle ResNet50V1.5的详细测试报告，包括执行环境、Paddle版本、环境搭建方法、复现脚本、测试结果和测试日志。

同时，给出了在同等执行环境下，业内几个知名框架在ResNet50V1.5模型下的性能数据，进行对比。

其他深度学习框架的 ResNet50V1.5 训练性能数据测试流程，请参考：[OtherReports](./OtherReports)。

<!-- omit in toc -->
## 目录
- [一、测试说明](#一测试说明)
- [二、环境介绍](#二环境介绍)
  - [1.物理机环境](#1物理机环境)
  - [2.Docker 镜像](#2docker-镜像)
- [三、环境搭建](#三环境搭建)
  - [1.单机（单卡、8卡）环境搭建](#1单机单卡8卡环境搭建)
  - [2.多机（32卡）环境搭建](#2多机32卡环境搭建)
- [四、测试步骤](#四测试步骤)
  - [1.单机（单卡、8卡）测试](#1单机单卡8卡测试)
  - [2.多机（32卡）测试](#2多机32卡测试)
- [五、测试结果](#五测试结果)
  - [1.Paddle训练性能](#1paddle训练性能)
  - [2.与业内其它框架对比](#2与业内其它框架对比)
- [六、日志数据](#六日志数据)

## 一、测试说明

我们统一使用 **吞吐能力** 作为衡量性能的数据指标。**吞吐能力** 是业界公认的、最主流的框架性能考核指标，它直接体现了框架训练的速度。

Resnet50V1.5 作为计算机视觉领域极具代表性的模型。在测试性能时，我们以 **单位时间内能够完成训练的图片数量（images/sec）** 作为训练期间的吞吐性能。在其它框架中，默认也均采用相同的计算方式。

测试中，我们选择如下3个维度，测试吞吐性能：

- **卡数**

   本次测试关注1卡、8卡、32卡情况下，模型的训练吞吐性能。选择的物理机是单机8卡配置。
   因此，1卡、8卡测试在单机下完成。32卡在4台机器下完成。

- **FP32/AMP**

   FP32 和 AMP 是业界框架均支持的两种精度训练模式，也是衡量框架性能的混合精度量化训练的重要维度。
   本次测试分别对 FP32 和 AMP 两种精度模式进行了测试。

- **BatchSize**

   本次测试，测试了BatchSize=128和BatchSize=256时，模型的吞吐率。BatchSize=128和BatchSize=256是业内最常使用的两种BatchSize大小。

关于其它一些参数的说明：
- **DALI**

   DALI 能够提升数据加载的速度，防止数据加载成为训练的瓶颈。因此，本次测试全部在打开 DALI 模式下进行。

- **XLA**

   本次测试的原则是测试 Resnet50V1.5 在 Paddle 下的最好性能表现，同时对比其与其它框架最好性能表现的优劣。

   因此，对于支持 XLA 的框架，我们默认打开 XLA 模式，已获得该框架最好的吞吐性能数据。

## 二、环境介绍
### 1.物理机环境

- 系统：CentOS Linux release 7.5.1804
- GPU：Tesla V100-SXM2-32GB * 8
- CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz * 40
- CUDA：11
- cuDNN：8.0.4
- 内存：502 GB

### 2.Docker 镜像

多数框架提供了包含完整测试环境的docker images，如下是各框架的基础环境配置：

> TODO(wanghuancoder):<br>
> 1. 协调益群、分布式确定docker环境配置,如：CUDA、Python、Paddle、DALI等等
> 2. 协调田硕制作镜像
> 3. 检查镜像制作是否满足测试需求

> TODO(Distribute):<br>
> 提供分布式训练，所需的环境配置、环境安装方法。可以不太详细，保证田硕能够安装正确即可。实际测试需要使用这个Docker进行测试，所以可以检验田硕的docker制作是否满足要求。

> TODO(wanghuancoder):<br>
> 完成以下信息整理

|配置 | Paddle | NGC TensorFlow | NGC PyTorch | NGC MxNet|
|-----|-----|-----|-----|-----|
| 框架版本 | 2.0 | 1.15.2 | 1.6.0a0+9907a3e | 1.5.0 |
| docker镜像 |  TODO hub.baidubce.com/paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82 | nvcr.io/nvidia/tensorflow:20.06-tf1-py3 | nvcr.io/nvidia/pytorch:20.07-py3 | nvcr.io/nvidia/mxnet:19.07-py3 |
| 模型代码 |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)|[NVIDIA/DeepLearningExamples/TensorFLow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5)|[NVIDIA/DeepLearningExamples/PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5)|[NVIDIA/DeepLearningExamples/MxNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5)|
|CUDA | 10.1 | 11 | 11 | 10.1 |
|cuDNN | 7.6.5 | 8.0.1 | 8.0.1 | 7.6.1 |

## 三、环境搭建

### 1.单机（单卡、8卡）环境搭建

> TODO(wanghuancoder):<br>
> 完成以下信息整理

- 安装docker
```
docker pull xxxx
```

- 启动docker
```
nvidia-docker ...
```

- 下载数据


### 2.多机（32卡）环境搭建

> TODO(Distribute):<br>
> 1. 提供分布式测试环境搭建的详细方法，可参考OneFlow的报告：<br>
> https://github.com/Oneflow-Inc/DLPerf/tree/master/PaddlePaddle/resnet50v1.5#nccl <br>
> https://github.com/Oneflow-Inc/DLPerf/tree/master/NVIDIADeepLearningExamples/TensorFlow/Classification/ConvNets/resnet50v1.5#ssh%E9%85%8D%E7%BD%AE%E5%8F%AF%E9%80%89 <br>
> 2. 注意：咱们Paddle也计划制作Docker镜像，将必要的环境安装在镜像中，如果分布式的环境搭建可以预安装到Docker中，请分布式同学联系王欢，共同制作Docker。而能够在Docker中预安装好的环境，可以在文档的环境搭建介绍中不提供具体安装方法。

- 多机网络部署

- 数据部署

## 四、测试步骤

### 1.单机（单卡、8卡）测试

> TODO(wanghuancoder):<br>
> 1. 编写一键测试脚本
> 2. 完成以下复现说明编写

请参考如下脚本搭建环境：
```
# 1. 安装docker

# 2. 启动docker

# 3. 下载数据

# 4. 执行测试脚本
```
执行测试脚本后，即可获得性能数据，如下图所示：

### 2.多机（32卡）测试

> TODO(Distribute):<br>
> 1. 使用PaddleClas中的Resnet50测试32卡分布式性能数据。
> 2. 编写一键执行的测试脚本，可参考： <br>
> https://github.com/Oneflow-Inc/DLPerf#benchmark-test-scopes <br>
> https://github.com/Oneflow-Inc/DLPerf#benchmark-test-scopes <br>

## 五、测试结果

### 1.Paddle训练性能
- 训练吞吐率(images/sec)如下:

|卡数 | FP32(BS=128) | FP32(BS=256) | AMP(BS=128) | AMP(BS=256)|
|-----|-----|-----|-----|-----|
|1 | - | - | - | -|
|8 | - | - | - | -|
|32 | - | - | - | -|

> TODO(wanghuancoder):<br>
> 完成测试，将1卡、8卡数据填入表格

> TODO(Distribute):<br>
> 完成测试，将32卡数据填入表格

### 2.与业内其它框架对比

说明：
- 同等执行环境下测试
- 单位：`images/sec`
- 对于支持 `DALI/XLA` 的框架，以下测试为开启 `DALI/XLA` 的数据

| 参数 | PaddlePaddle | NGC TensorFlow 1.15 | NGC PyTorch | NGC MXNet |
|-----|-----|-----|-----|-----|
| FP32 GPU=1,BS=128 | - | 398.257 | 355.69 | 376.18 |
| AMP GPU=1,BS=256 | - | 985.466 | 797.38 | 1398.8 |
| FP32 GPU=8,BS=128 | - | - | - | - |
| AMP GPU=8,BS=256 | - | - | - | - |
| FP32 GPU=32,BS=128 | - | - | - | - |
| AMP GPU=32,BS=256 | - | - | - | - |

> TODO(wanghuancoder):<br>
> 完成测试，将1卡、8卡数据填入表格

> TODO(Distribute):<br>
> 完成测试，将32卡数据填入表格

## 六、日志数据
- [1卡 FP32 BS=128 日志](./logs/)
- [1卡 FP32 BS=160 日志](./logs/)
- ...

> TODO(wanghuancoder):<br>
> 完成测试，将1卡、8卡 与 公布性能数据 一致的原始日志文件提交到log目录下，并更新链接

> TODO(Distribute):<br>
> 完成测试，将32卡 与 公布性能数据 一致的原始日志文件提交到log目录下，并更新链接
