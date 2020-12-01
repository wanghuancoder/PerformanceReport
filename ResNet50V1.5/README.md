# Paddle ResNet50V1.5 性能测试报告

这里给出了Paddle ResNet50V1.5的测试报告，包括执行环境、Paddle版本、环境搭建方法、复现脚本、测试结果和测试日志。

同时，给出了在同等执行环境下，业内几个知名框架在ResNet50V1.5模型下的性能数据，和测试报告。

## 执行环境

- GPU：NVIDIA V100-SXM2-16GB
- CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
- 内存：64 GB
- 系统：Ubuntu 18.04.4 LTS
- CUDA：11
- cuDNN：8.0.4

## Docker环境

Docker使用Paddle官方公布的 `paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82`（TODO）
其软件配置如下：
- Paddle 2.0.0
- models：
- 系统：Ubuntu 18.04.4 LTS
- CUDA：11
- cuDNN：8.0.4
- python 3.7

- NCCL 2.6.3
- Horovod 0.19.0
- OpenMPI 3.1.4
- DALI 0.19.0

## 测试说明

- Node
- BatchSize
- FP32/AMP
- XLA

## 环境搭建&测试

请参考如下脚本搭建环境：
```
# 1. 安装docker

# 2. 启动docker

# 3. 下载数据

# 4. 执行测试脚本
```
执行测试脚本后，即可获得性能数据，如下图所示：

## 测试结果

## 与业内其它框架的数据对比

| 参数 | PaddlePaddle | NGC TensorFlow 1.15 | NGC PyTorch | NGC MXNet | OneFlow |
|-----|-----|-----|-----|-----|-----|
| FP32 GPU=1,BS=128 | - | - | - | - | - |
| AMP GPU=1,BS=256 | - | - | - | - | - |
| FP32 GPU=8,BS=128 | - | - | - | - | - |
| AMP GPU=8,BS=256 | - | - | - | - | - |
| FP32 GPU=32,BS=128 | - | - | - | - | - |
| AMP GPU=32,BS=256 | - | - | - | - | - |
