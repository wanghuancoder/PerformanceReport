# Paddle Performance Report——Paddle框架性能测试报告

## 介绍

本repo用于公开PaddlePaddle重点模型，在训练期间的性能数据，以供业界参考。

同时，本repo也尝试在同等执行环境下，复现了业内部分其它知名深度学习框架对应模型的性能数据，以供对比。

## 测试模型

- [ResNet50V1.5](./ResNet50V1.5)

我们将持续开展性能测试工作，后续将有更多性能数据，逐步公开。

## 供对比的业内深度学习框架

我们选择了NGC优化后的TensorFlow、PyTorch、MxNet，以及国内优秀深度学习框架OneFlow，作为性能对比参照。
对这些框架的性能测试，我们选用相同的物理机执行，并严格参照各框架官网公布的测试方法执行。

- [NGC TensorFlow 1.15](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags) 
- [NGC PyTorch](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch/tags)
- [NGC MxNet](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet/tags)
- [OneFlow](https://github.com/Oneflow-Inc/oneflow/tree/v0.2.0)

## 测试结果

### ResNet50V1.5
- GPU型号：V100-SXM2-16GB
- 单位：images/sec
- 对于支持DALI/XLA的框架，以下测试为开启DALI/XLA的数据

| 参数 | PaddlePaddle | NGC TensorFlow 1.15 | NGC PyTorch | NGC MXNet | OneFlow |
|-----|-----|-----|-----|-----|-----|
| FP32 GPU=1,BS=128 | - | - | - | - | - |
| AMP GPU=1,BS=256 | - | - | - | - | - |
| FP32 GPU=8,BS=128 | - | - | - | - | - |
| AMP GPU=8,BS=256 | - | - | - | - | - |
| FP32 GPU=32,BS=128 | - | - | - | - | - |
| AMP GPU=32,BS=256 | - | - | - | - | - |
