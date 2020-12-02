# Paddle Performance Report——Paddle框架性能测试报告


本 repo 用于公开 PaddlePaddle 开源实现的各个学术界、工业界各前沿模型，在训练期间的性能数据，同时提供了各模型性能测试的详细复现流程，以供参考。

同时，我们也在相同的硬件执行环境下，按照业内其它知名深度学习框架公开的代码和教程，测试了相同模型的性能数据。

## 目录
- [测试模型](#测试模型)
    * [ResNet50V1.5](./ResNet50V1.5)
    * [Bert Base Pre-Training](./Bert)

- [供对比的业内深度学习框架](#供对比的业内深度学习框架)
    * [NGC TensorFlow 1.15](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags)
    * [NGC PyTorch](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch/tags)
    * [NGC MxNet](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet/tags)
    * [OneFlow](https://github.com/Oneflow-Inc/oneflow/tree/v0.2.0)

- [测试结果](#测试结果)
    * [ResNet50V1.5](#ResNet50V1.5)
    * [Bert Base Pre-Training](#Bert-Base-Pre-Training)

## 测试模型

- [ResNet50V1.5](./ResNet50V1.5)
- [Bert Base Pre-Training](./Bert)

我们将持续开展性能测试工作，后续将逐步公开更多性能数据。

## 供对比的业内深度学习框架

我们选择了 NGC 优化后的 TensorFlow 、PyTorch、MxNet，以及国内优秀的深度学习框架 OneFlow，作为性能对比参照。

对这些框架的性能测试，我们选用相同的物理机执行，并严格参照各框架官网公布的测试方法执行。

- [NGC TensorFlow 1.15](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags)
- [NGC PyTorch](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch/tags)
- [NGC MxNet](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet/tags)
- [OneFlow](https://github.com/Oneflow-Inc/oneflow/tree/v0.2.0)

## 测试结果

说明：

- GPU型号：`V100-SXM2-16GB`
- 单位：`images/sec`
- 对于支持 `DALI/XLA` 的框架，以下测试为开启 `DALI/XLA` 的数据
- BatchSize 选用各框架支持的最大 BatchSize（下简称：BS）

### ResNet50V1.5


| 参数 | PaddlePaddle | NGC TensorFlow 1.15 | NGC PyTorch | NGC MXNet | OneFlow |
|-----|-----|-----|-----|-----|-----|
| FP32 GPU=1,BS=max | - | - | - | - | - |
| AMP GPU=1,BS=max | - | - | - | - | - |
| FP32 GPU=8,BS=max | - | - | - | - | - |
| AMP GPU=8,BS=max | - | - | - | - | - |
| FP32 GPU=32,BS=max | - | - | - | - | - |
| AMP GPU=32,BS=max | - | - | - | - | - |


### Bert Base Pre-Training


| 参数 | PaddlePaddle | NGC TensorFlow 1.15 | NGC PyTorch | OneFlow |
|-----|-----|-----|-----|-----|-----|
| FP32 GPU=1,BS=max | - | - | - | - | - |
| AMP GPU=1,BS=max | - | - | - | - | - |
| FP32 GPU=8,BS=max | - | - | - | - | - |
| AMP GPU=8,BS=max | - | - | - | - | - |
| FP32 GPU=32,BS=max | - | - | - | - | - |
| AMP GPU=32,BS=max | - | - | - | - | - |
