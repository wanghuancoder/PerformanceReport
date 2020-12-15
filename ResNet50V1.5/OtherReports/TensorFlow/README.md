# NGC TensorFlow ResNet50V1.5 性能测试

此处给出了基于 [NGC TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5) 实现的 ResNet50V1.5 任务的详细复现流程，包括环境介绍、环境搭建、复现脚本、测试结果和测试日志等。

<!-- omit in toc -->
## 目录
- [NGC TensorFlow ResNet50V1.5 性能测试](#ngc-tensorflow-resnet50v15-性能测试)
  - [一、环境介绍](#一环境介绍)
    - [1.物理机环境](#1物理机环境)
    - [2.Docker 镜像](#2docker-镜像)
  - [二、环境搭建](#二环境搭建)
    - [1.单机（单卡、8卡）环境搭建](#1单机单卡8卡环境搭建)
    - [2.多机（32卡）环境搭建](#2多机32卡环境搭建)
  - [三、测试步骤](#三测试步骤)
    - [1.单机（单卡、8卡）测试](#1单机单卡8卡测试)
    - [2.多机（32卡）测试](#2多机32卡测试)
  - [四、测试结果](#四测试结果)
  - [五、日志数据](#五日志数据)

## 一、环境介绍

### 1.物理机环境

我们使用了与Paddle测试完全相同的物理机环境：

- 系统：CentOS Linux release 7.5.1804
- GPU：Tesla V100-SXM2-32GB * 8
- CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz * 40
- CUDA：11
- cuDNN：8.0.4
- Driver Version: 450.80.02
- 内存：502 GB

### 2.Docker 镜像

我们使用 NGC TensorFlow 的代码仓库提供的Dockerfile制作镜像：

- Docker: nvcr.io/nvidia/tensorflow:20.06-tf1-py3
- TensorFlow：1.15.2
- 模型代码：[NVIDIA/DeepLearningExamples/TensorFLow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5)
- CUDA：11
- cuDNN：8.0.1

## 二、环境搭建

### 1.单机（单卡、8卡）环境搭建

单机环境的搭建，我们遵循了 NGC TensorFlow 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5#quick-start-guide) 教程成功搭建了测试环境，主要过程如下：

- 以ImageNet2012数据集为基础制作TF_Record格式的数据

这部分不在本报告中详细展开，可参考NGC提供的[文档](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5#quick-start-guide)制作。

- 下载NGC TensorFlow repo,并进入目录
   ```bash
   git clone https://github.com/NVIDIA/DeepLearningExamples
   cd DeepLearningExamples/TensorFlow/Classification/ConvNets
   # 本次测试是在如下版本下完成的：
   git checkout 99b1c898cead5603c945721162270c2fe077b4a2
   ```

- 制作Docker镜像
   ```bash
   docker build . -t nvidia_rn50_tf
   ```

- 启动Docker
   ```bash
   # 假设制作好的TF_Record数据放在<path to tfrecords data>目录下
   nvidia-docker run --rm -it -v <path to tfrecords data>:/data/tfrecords --ipc=host nvidia_rn50_tf
   ```

### 2.多机（32卡）环境搭建

对于32卡性能测试，由于NGC并未提供测试环境和测试方法，我们参考[XXX]()搭建了测试环境，完成了测试。
> TODO(Distribute):<br>
> 找到一个TF或NGC官方的、关于分布式使用的文档，放在XXX位置，并提供链接。后续的测试，也真正参考这个文档进行测试。


> TODO(Distribute):<br>
> 1. 提供分布式测试环境搭建的详细方法，这里最好先确定是否能够使用NGC提供的官方docker `NGC 20.03`，完成分布式测试。否则，需要详细给出环境的搭建方法。参考材料如下： <br>
> https://github.com/Oneflow-Inc/DLPerf/tree/master/PaddlePaddle/resnet50v1.5#nccl <br>
> https://github.com/Oneflow-Inc/DLPerf/tree/master/NVIDIADeepLearningExamples/TensorFlow/Classification/ConvNets/resnet50v1.5#ssh%E9%85%8D%E7%BD%AE%E5%8F%AF%E9%80%89 <br>

- SSH
- BI
- IP

## 三、测试步骤

### 1.单机（单卡、8卡）测试

对于1卡、8卡性能测试，本报告严格按NGC公开的测试报告进行复现，对其提供的代码未做改动，并严格按照NGC测试使用的参数配置测试。其公开的测试报告请见：[《ResNet-50 v1.5 for TensorFlow》](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5)

- 下载我们编写的测试脚本，并执行该脚本
   ```bash
   wget https://raw.githubusercontent.com/wanghuancoder/PerformanceReport/main/ResNet50V1.5/OtherReports/TensorFlow/scripts/tf_test_all.sh
   bash tf_test_all.sh
   ```

- 执行后将得到如下日志文件：
   ```bash
   /data/tfrecords/log/tf_gpu1_fp32_bs128.txt
   /data/tfrecords/log/tf_gpu1_fp32_bs256.txt
   /data/tfrecords/log/tf_gpu1_amp_bs128.txt
   /data/tfrecords/log/tf_gpu1_amp_bs256.txt
   /data/tfrecords/log/tf_gpu8_fp32_bs128.txt
   /data/tfrecords/log/tf_gpu8_fp32_bs256.txt
   /data/tfrecords/log/tf_gpu8_amp_bs128.txt
   /data/tfrecords/log/tf_gpu8_amp_bs256.txt
   ```

由于NGC TensorFlow的测试使用的是`training_perf.sh`，因此我们提供的`tf_test_all.sh`是参考了`training_perf.sh`的参数设置方法。

### 2.多机（32卡）测试

> TODO(Distribute):<br>
> 1. 之后，进行测试，给出测试脚本。测试脚本最好是一键执行的。可参考NGC提供的`resnet50v1.5/training/training_perf.sh`脚本。也可以参考： <br>
> https://github.com/Oneflow-Inc/DLPerf#benchmark-test-scopes <br>
> https://github.com/Oneflow-Inc/DLPerf#benchmark-test-scopes <br>
> 2. 得出32卡数据，并保留日志文件。
> 3. 最好能够找到NGC/TF官方公布的多卡性能数据做对比，原则上我们复现的性能数据该与对方公布的数据基本接近，否则应认真检查我们的复现是否存在问题。OneFlow公开的TF 32卡数据参考如下： <br>
> https://github.com/Oneflow-Inc/DLPerf/tree/master/NVIDIADeepLearningExamples/TensorFlow/Classification/ConvNets/resnet50v1.5#resnet50-v15-fp32  <br>

请参考如下脚本搭建环境：
```
```

## 四、测试结果

- 训练吞吐率(images/sec)如下:

|卡数 | FP32(BS=128) | FP32(BS=256) | AMP(BS=128) | AMP(BS=256)|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|1 | 398.349 | 414.631 | 979.091 | 994.131 |
|8 | 3038.730 | 3210.793 | 7601.888 | 7799.018 |
|32 | - | - | - | -|

> TODO(Distribute):<br>
> 完成测试，将32卡数据填入表格

## 五、日志数据
- [1卡 FP32 BS=128 日志](./logs/tf_gpu1_fp32_bs128.txt)
- [1卡 FP32 BS=256 日志](./logs/tf_gpu1_fp32_bs256.txt)
- [1卡 AMP BS=128 日志](./logs/tf_gpu1_amp_bs128.txt)
- [1卡 AMP BS=256 日志](./logs/tf_gpu1_amp_bs256.txt)
- [8卡 FP32 BS=128 日志](./logs/tf_gpu8_fp32_bs128.txt)
- [8卡 FP32 BS=256 日志](./logs/tf_gpu8_fp32_bs256.txt)
- [8卡 AMP BS=128 日志](./logs/tf_gpu8_amp_bs128.txt)
- [8卡 AMP BS=256 日志](./logs/tf_gpu8_amp_bs256.txt)

> TODO(Distribute):<br>
> 完成测试，将32卡 与 公布性能数据 一致的原始日志文件提交到log目录下，并更新链接
