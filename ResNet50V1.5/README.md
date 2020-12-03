# Paddle ResNet50V1.5 性能测试

此处给出了Paddle ResNet50V1.5的详细测试报告，包括执行环境、Paddle版本、环境搭建方法、复现脚本、测试结果和测试日志。

同时，给出了在同等执行环境下，业内几个知名框架在ResNet50V1.5模型下的性能数据，和测试报告。

## 目录
- [一、环境介绍](#一环境介绍)
    * [1. 物理机环境](#1物理机环境)
    * [2. Docker 镜像](#2Docker镜像)
- [二、测试说明](#二测试说明)
- [三、环境搭建](#三环境搭建)
- [四、测试步骤](#四测试步骤)
    * [1. 单机（单卡、8卡）测试](#1单机单卡8卡测试)
    * [2. 多机（32卡）测试](#2多机32卡测试)
- [五、测试结果](#四测试结果)
    * [1. Paddle训练性能](#1Paddle训练性能)
    * [2. 与业内其它框架对比](#2与业内其它框架对比)
- [六、日志数据](#五日志数据)

## 一、环境介绍
### 1.物理机环境

- 系统：Ubuntu 18.04.4 LTS
- GPU：NVIDIA V100-SXM2-16GB
- CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz * 38
- CUDA：11
- cuDNN：8.0.4
- 内存：448 GB

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

|配置 | Paddle | NGC TensorFlow | NGC PyTorch | OneFlow|
|-----|-----|-----|-----|-----|
| 框架版本 | 2.0 | 1.15.2+nv | 1.6.0a0+9907a3e | 0.2.0 |
| docker镜像 |  paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82 | - | - | - |
| 模型代码 | | | |
|CUDA |  |  |  |  |
|cuDNN | 8.0.4 | 8.0.1 | 8.0.1 | - |


## 二、测试说明

> TODO(wanghuancoder):<br>
> 完成以下信息整理

- Node
- BatchSize
- FP32/AMP
- XLA

网络结构：

优化器：

## 三、环境搭建

### 1.单机（单卡、8卡）测试

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


### 2.多机（32卡）测试

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

|卡数 | FP32(BS=128) | FP32(BS=160) | AMP(BS=128) | AMP(BS=208)|
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
- BatchSize 选用各框架支持的最大 BatchSize

| 参数 | PaddlePaddle | NGC TensorFlow 1.15 | NGC PyTorch | NGC MXNet | OneFlow |
|-----|-----|-----|-----|-----|-----|
| FP32 GPU=1,BS=max | - | - | - | - | - |
| AMP GPU=1,BS=max | - | - | - | - | - |
| FP32 GPU=8,BS=max | - | - | - | - | - |
| AMP GPU=8,BS=max | - | - | - | - | - |
| FP32 GPU=32,BS=max | - | - | - | - | - |
| AMP GPU=32,BS=max | - | - | - | - | - |

> TODO(wanghuancoder):<br>
> 完成测试，将1卡、8卡数据填入表格

> TODO(Distribute):<br>
> 完成测试，将32卡数据填入表格

## 六、日志数据
- [1卡 FP32 BS=128 日志](./log/)
- [1卡 FP32 BS=160 日志](./log/)
- ...

> TODO(wanghuancoder):<br>
> 完成测试，将1卡、8卡 与 公布性能数据 一致的原始日志文件提交到log目录下，并更新链接

> TODO(Distribute):<br>
> 完成测试，将32卡 与 公布性能数据 一致的原始日志文件提交到log目录下，并更新链接
