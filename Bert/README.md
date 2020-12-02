# Paddle Bert Base 性能测试报告

这里给出了基于Paddle实现 Bert Base Pre-Training 任务的详细测试报告，包括执行环境、Paddle版本、环境搭建方法、复现脚本、测试结果和测试日志。

同时，给出了在相同执行环境下，业内几个知名框架的性能数据。


## 执行环境

- 系统：Ubuntu 18.04.4 LTS
- GPU：NVIDIA V100-SXM2-16GB
- CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
- CUDA：11
- cuDNN：8.0.4
- 内存：64 GB

多数框架提供了包含完整测试环境的docker images，如下是各框架的基础环境配置：

|配置 | Paddle | NGC TensorFlow | NGC PyTorch | OneFlow|
|-----|-----|-----|-----|-----|
| 框架版本 | 2.0 | 1.15.2+nv | 1.6.0a0+9907a3e | 0.2.0 |
| docker镜像 |  paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82   | nvcr.io/nvidia/tritonserver:20.09-py3 | nvcr.io/nvidia/pytorch:20.06-py3 | -
|模型代码| | [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) | [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)| [Oneflow-Inc/OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/637bb9cdb4cc1582f13bcc171acbc8a8089d9435/LanguageModeling/BERT) |
|CUDA | 11 | 11 | 11 | 11 |
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

- 训练吞吐率(sentences/sec)如下:

|卡数 | FP32(BS=32) | AMP(BS=64) | FP32(BS=max) | AMP(BS=max) |
|-----|-----|-----|-----|-----|
|1 | - | - | - | - |
|8 | - | - | - | - |
|32 | - | - | - | - |


## 与业内其它框架的数据对比

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

# 日志数据