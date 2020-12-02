# Paddle Bert Base 性能测试

此处给出了基于 Paddle 框架实现的 Bert Base Pre-Training 任务的详细测试报告，包括执行环境、Paddle版本、环境搭建、复现脚本、测试结果和测试日志。

相同环境下，其他深度学习框架的 Bert 训练性能数据测试流程，请参考：[OtherReports](./OtherReports)。

## 目录
- [一、环境介绍](#一、环境介绍)
    * [1. 物理机环境](#1.物理机环境)
    * [2. Docker 镜像](#2.Docker-镜像)
- [二、环境搭建](#二、环境搭建)
- [三、测试步骤](#三、测试步骤)
    * [1. 单机（单卡、8卡）测试](#1.单机（单卡、8卡）测试)
    * [2. 多机（32卡）测试](#2.多机（32卡）测试)
- [四、测试结果](#四、测试结果)
    * [1. Paddle训练性能](#1.Paddle训练性能)
    * [2. 与业内其它框架对比](#2.与业内其它框架对比)
- [五、日志数据](#五、日志数据)

## 一、环境介绍
### 1.物理机环境

- 系统：Ubuntu 18.04.4 LTS
- GPU：NVIDIA V100-SXM2-16GB
- CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
- CUDA：11
- cuDNN：8.0.4
- 内存：64 GB

### 2.Docker 镜像

多数框架提供了包含完整测试环境的docker images，如下是各框架的基础环境配置：

|配置 | Paddle | NGC TensorFlow | NGC PyTorch | OneFlow|
|-----|-----|-----|-----|-----|
| 框架版本 | 2.0 | 1.15.2+nv | 1.6.0a0+9907a3e | 0.2.0 |
| docker镜像 |  paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82   | nvcr.io/nvidia/tritonserver:20.09-py3 | nvcr.io/nvidia/pytorch:20.06-py3 | -
|模型代码| | [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) | [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)| [Oneflow-Inc/OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/637bb9cdb4cc1582f13bcc171acbc8a8089d9435/LanguageModeling/BERT) |
|CUDA | 11 | 11 | 11 | 11 |
|cuDNN | 8.0.4 | 8.0.1 | 8.0.1 | - |

## 二、环境搭建

各深度学习框架在公开的 Github 仓库中给出了详细的docker镜像和构建脚本，具体搭建流程请参考：[此处](./OtherReports)。

如下是 Paddle 测试环境的具体搭建流程:

1. 安装docker

```bash
docker pull paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82
```

2. 启动docker
```bash
nvidia-docker ...
```

3. 下载数据

Bert 模型的 Pre-Training 任务是基于 [wikipedia]() 和 [BookCorpus]() 数据集进行的训练的，原始数据集比较大。我们提供了一份小的、且已处理好的[样本数据集]()，可以下载并解压到`XX`目录里。


> TODO(分布式):<br>
> 1. 若多机多卡环境与单机不公用，请**务必**给出详细的多机环境配置流程<br>
> 2. 可以单列一个三级标题  **### 多机环境搭建** 来介绍

## 三、测试步骤

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

## 四、测试结果

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
|-----|-----|-----|-----|-----|-----|
| FP32 GPU=1,BS=max | - | - | - | - | - |
| AMP GPU=1,BS=max | - | - | - | - | - |
| FP32 GPU=8,BS=max | - | - | - | - | - |
| AMP GPU=8,BS=max | - | - | - | - | - |
| FP32 GPU=32,BS=max | - | - | - | - | - |
| AMP GPU=32,BS=max | - | - | - | - | - |

## 五、日志数据
