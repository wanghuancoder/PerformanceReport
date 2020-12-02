# Paddle ResNet50V1.5 性能测试报告

这里给出了Paddle ResNet50V1.5的测试报告，包括执行环境、Paddle版本、环境搭建方法、复现脚本、测试结果和测试日志。

同时，给出了在同等执行环境下，业内几个知名框架在ResNet50V1.5模型下的性能数据，和测试报告。

## 执行环境

- 系统：Ubuntu 18.04.4 LTS
- GPU：NVIDIA V100-SXM2-16GB
- CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz * 38
- CUDA：11
- cuDNN：8.0.4
- 内存：448 GB


多数框架提供了包含完整测试环境的docker images，如下是各框架的基础环境配置：

|配置 | Paddle | NGC TensorFlow | NGC PyTorch | OneFlow|
|-----|-----|-----|-----|-----|
| 框架版本 | 2.0 | 1.15.2+nv | 1.6.0a0+9907a3e | 0.2.0 |
| docker镜像 |  paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82 | - | - | - |
| 模型代码 | | | |
|CUDA |  |  |  |  |
|cuDNN | 8.0.4 | 8.0.1 | 8.0.1 | - |

## 环境搭建&测试

请参考如下脚本搭建环境：
```
# 1. 安装docker

# 2. 启动docker

# 3. 下载数据

# 4. 执行测试脚本
```
执行测试脚本后，即可获得性能数据，如下图所示：


## 测试说明

- Node
- BatchSize
- FP32/AMP
- XLA

网络结构：

优化器：

## 测试结果

- 训练吞吐率(images/sec)如下:

|卡数 | FP32(BS=128) | FP32(BS=160) | AMP(BS=128) | AMP(BS=208)|
|-----|-----|-----|-----|-----|
|1 | - | - | - | -|
|8 | - | - | - | -|
|32 | - | - | - | -|

多卡加速比如下：

|卡数 | FP32(BS=128) | FP32(BS=160) | AMP(BS=128) | AMP(BS=208)|
|-----|-----|-----|-----|-----|
|1 | - | - | - | -|
|8 | - | - | - | -|
|32 | - | - | - | -|

## 与业内其它框架的数据对比

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

# 日志数据
