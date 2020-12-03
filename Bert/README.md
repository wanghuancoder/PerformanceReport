# Paddle Bert Base 性能测试

此处给出了基于 Paddle 框架实现的 Bert Base Pre-Training 任务的训练性能详细测试报告，包括执行环境、Paddle 版本、环境搭建、复现脚本、测试结果和测试日志。

相同环境下，其他深度学习框架的 Bert 训练性能数据测试流程，请参考：[OtherReports](./OtherReports)。

## 目录
- [一、测试说明](#一测试说明)
- [二、环境介绍](#二环境介绍)
    * [1. 物理机环境](#1物理机环境)
    * [2. Docker 镜像](#2Docker镜像)
- [三、环境搭建](#三环境搭建)
    * [1.单机（单卡、8卡）环境搭建](#1单机单卡8卡环境搭建)
    * [2.多机（32卡）环境搭建](#2多机32卡环境搭建)
- [四、测试步骤](#四测试步骤)
    * [1. 单机（单卡、8卡）测试](#1单机单卡8卡测试)
    * [2. 多机（32卡）测试](#2多机32卡测试)
- [五、测试结果](#五测试结果)
    * [1. Paddle训练性能](#1Paddle训练性能)
    * [2. 与业内其它框架对比](#2与业内其它框架对比)
- [六、日志数据](#六日志数据)



## 一、测试说明

我们统一使用了 **吞吐能力** 作为衡量性能的数据指标。**吞吐能力** 是业界公认的、最主流的框架性能考核指标，它直接体现了框架训练的速度。

Bert Base 模型是自研语言处理领域极具代表性的模型，包括 Pre-Training 和 Fine-tune 两个子任务，此处我们选取 Pre-Training 阶段作为测试目标。在测试性能时，我们以 **sentences/sec）** 作为训练期间的吞吐性能。在其它框架中，默认也均采用相同的计算方式。

测试中，我们选择如下3个维度，测试吞吐性能：

- 卡数

   本次测试关注1卡、8卡、32卡情况下，模型的训练吞吐性能。选择的物理机是单机8卡配置。
   因此，1卡、8卡测试在单机下完成。32卡在4台机器下完成。

- FP32/AMP

   FP32 和 AMP 是业界框架均支持的两种精度训练模式，也是衡量框架性能的混合精度量化训练的重要维度。
   本次测试分别对 FP32 和 AMP 两种精度模式进行了测试。


- BatchSize

   经调研，大多框架的 Bert Base Pre-Training 任务在第一阶段 max_seq_len=128的数据集训练时 ，均支持 FP32 模式下 BatchSize=32，AMP 模式下 BatchSize=64。因此我们分别测试了上述两种组合方式下的吞吐性能。

   此外，还测试了 Bert Base 在各框架 FP32、AMP 精度模式下支持的最大BatchSize，及对应的吞吐性能，这通常也代表了其最好的性能表现。

关于其它一些参数的说明：

- XLA

   本次测试的原则是测试 Bert Base 在 Paddle 下的最好性能表现，同时对比其与其它框架最好性能表现的优劣。

   因此，对于支持 XLA 的框架，我们默认打开 XLA 模式，已获得该框架最好的吞吐性能数据。

- 优化器

   > TODO(Aurelius84): 最终确认 Paddle 使用的优化器类型
   在 Bert Base 的 Pre-Training 任务上，各个框架使用的优化器略有不同。NGC TensorFlow、NGC PyTorch、PaddlePaddle 均支持 LAMBOptimizer，OneFlow默认仅支持了AdamOptimizer。

   此处我们以各个框架默认使用的优化器为准，并测试模型的吞吐性。

## 二、环境介绍
### 1.物理机环境

- 系统：Ubuntu 18.04.4 LTS
- GPU：NVIDIA V100-SXM2-16GB
- CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz * 38
- CUDA：11
- cuDNN：8.0.4
- 内存：448 GB

### 2.Docker 镜像
> TODO(Aurelius84): 待更新Paddle开源出去的docker镜像tags

多数框架提供了包含完整测试环境的docker images，如下是各框架的基础环境配置：

|配置 | Paddle | NGC TensorFlow | NGC PyTorch | OneFlow|
|-----|-----|-----|-----|-----|
| 框架版本 | 2.0 | 1.15.2+nv | 1.6.0a0+9907a3e | 0.2.0 |
| docker镜像 |  paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82   | nvcr.io/nvidia/tritonserver:20.09-py3 | nvcr.io/nvidia/pytorch:20.06-py3 | -
|模型代码| | [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) | [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)| [Oneflow-Inc/OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/637bb9cdb4cc1582f13bcc171acbc8a8089d9435/LanguageModeling/BERT) |
|CUDA | 11 | 11 | 11 | 11 |
|cuDNN | 8.0.4 | 8.0.1 | 8.0.1 | - |

## 三、环境搭建

各深度学习框架在公开的 Github 仓库中给出了详细的docker镜像和构建脚本，具体搭建流程请参考：[此处](./OtherReports)。

如下是 Paddle 测试环境的具体搭建流程:

### 1.单机（单卡、8卡）环境搭建

1. 安装docker

```bash
docker pull paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82
```

2. 启动docker
```bash
nvidia-docker ...
```

3. 下载数据

> TODO(Aurelius84): 待上传样本数据集，并给出下载链接和解压路径

Bert 模型的 Pre-Training 任务是基于 [wikipedia]() 和 [BookCorpus]() 数据集进行的训练的，原始数据集比较大。我们提供了一份小的、且已处理好的[样本数据集]()，可以下载并解压到`XX`目录里。


### 2.多机（32卡）环境搭建

> TODO(Distribute):<br>
> 1. 提供分布式测试环境搭建的详细方法，可参考OneFlow的报告：<br>
> https://github.com/Oneflow-Inc/DLPerf/tree/master/PaddlePaddle/bert#nccl <br>
> https://github.com/Oneflow-Inc/DLPerf/tree/master/PaddlePaddle/bert#2%E6%9C%BA16%E5%8D%A1 <br>
> 2. 注意：咱们Paddle也计划制作Docker镜像，将必要的环境安装在镜像中，如果分布式的环境搭建可以预安装到Docker中，请分布式同学联系王欢，共同制作Docker。而能够在Docker中预安装好的环境，可以在文档的环境搭建介绍中不提供具体安装方法。

## 四、测试步骤

### 1.单机（单卡、8卡）测试
> TODO(Aurelius84):(需包含)<br>
> 1. 单机单卡、8卡公用的可修改配置的同一个执行shell脚本，给出脚本文件链接<br>
> 2. 对重要参数进行逐一说明<br>
> 3. 给出单机单卡、8卡的执行命令


### 2.多机（32卡）测试
> TODO(分布式):(需包含)<br>
> 1. 提供多机32卡可修改配置的同一个执行shell脚本，给出脚本文件链接<br>
> 2. 对重要参数进行逐一说明<br>
> 3. 给出多机32卡的执行命令

## 五、测试结果

### 1.Paddle训练性能
- 训练吞吐率(sentences/sec)如下:

|卡数 | FP32(BS=32) | AMP(BS=64) | FP32(BS=max) | AMP(BS=max) |
|-----|-----|-----|-----|-----|
|1 | - | - | - | - |
|8 | - | - | - | - |
|32 | - | - | - | - |


### 2.与业内其它框架对比

说明：
- 同等执行环境下测试
- 单位：`sentences/sec`
- 对于支持 `DALI/XLA` 的框架，以下测试为开启 `DALI/XLA` 的数据
- BatchSize 选用各框架支持的最大 BatchSize

| 参数 | PaddlePaddle | NGC TensorFlow 1.15 | NGC PyTorch | OneFlow |
|-----|-----|-----|-----|-----|
| FP32 GPU=1,BS=max | - | - | - | - |
| AMP GPU=1,BS=max | - | - | - | - |
| FP32 GPU=8,BS=max | - | - | - | - |
| AMP GPU=8,BS=max | - | - | - | - |
| FP32 GPU=32,BS=max | - | - | - | - |
| AMP GPU=32,BS=max | - | - | - | - |

## 六、日志数据
