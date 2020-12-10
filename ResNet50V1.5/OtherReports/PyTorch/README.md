# NGC PyTorch ResNet50V1.5 性能测试

这里给出 NGC PyTorch ResNet50V1.5的性能测试报告。
对于1卡、8卡性能测试，本报告严格按NGC公开的测试报告进行复现，对其提供的代码未做改动，并严格按照NGC测试使用的参数配置测试。其公开的测试报告请见：[《ResNet50 v1.5 For PyTorch》](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5)

对于32卡性能测试，由于NGC并未提供测试环境和测试方法，我们参考[XXX]()搭建了测试环境，完成了测试。
> TODO(Distribute):<br>
> 找到一个TF或NGC官方的、关于分布式使用的文档，放在XXX位置，并提供链接。后续的测试，也真正参考这个文档进行测试。

## 目录
- [一、环境介绍](#一环境介绍)
- [二、环境搭建](#二环境搭建)
    * [1.单机（单卡、8卡）环境搭建](#1单机单卡8卡环境搭建)
    * [2.多机（32卡）环境搭建](#2多机32卡环境搭建)
- [三、测试步骤](#三测试步骤)
    * [1.单机（单卡、8卡）测试](#1单机单卡8卡测试)
    * [2.多机（32卡）测试](#2多机32卡测试)
- [四、测试结果](#四测试结果)
- [五、日志数据](#五日志数据)

## 一、环境介绍
环境介绍（物理机环境及Docker环境）在[《Paddle ResNet50V1.5 性能测试》](../../)中已经给出。

所有测试物理机环境完全一致，Docker环境使用NVIDIA官方提供的`nvcr.io/nvidia/pytorch:20.07-py3`镜像。

## 二、环境搭建

### 1.单机（单卡、8卡）环境搭建

- 下载NGC PyTorch repo,并进入目录
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/PyTorch/Classification/ConvNets
# 本次测试是在如下版本下完成的：
git checkout 99b1c898cead5603c945721162270c2fe077b4a2
```

- 制作Docker镜像
```
docker build . -t nvidia_rn50_pytorch
```

- 启动Docker
```
# 假设imagenet数据放在<path to data>目录下
nvidia-docker run --rm -it -v <path to data>:/imagenet --ipc=host nvidia_rn50_pytorch
```

### 2.多机（32卡）环境搭建

> TODO(Distribute):<br>
> 1. 提供分布式测试环境搭建的详细方法，这里最好先确定是否能够使用NGC提供的官方docker `NGC 20.03`，完成分布式测试。否则，需要详细给出环境的搭建方法。参考材料如下： <br>
> https://github.com/Oneflow-Inc/DLPerf/tree/master/PaddlePaddle/resnet50v1.5#nccl <br>
> https://github.com/Oneflow-Inc/DLPerf/tree/master/NVIDIADeepLearningExamples/TensorFlow/Classification/ConvNets/resnet50v1.5#ssh%E9%85%8D%E7%BD%AE%E5%8F%AF%E9%80%89 <br>

- SSH
- BI
- IP

## 三、测试步骤

### 1.单机（单卡、8卡）测试

- 下载我们编写的测试脚本，并执行该脚本
```
wget https://raw.githubusercontent.com/wanghuancoder/PerformanceReport/main/ResNet50V1.5/OtherReports/TensorFlow/scripts/test_all.sh
bash test_all.sh
```

- 执行后将得到如下日志文件：
```
/imagenet/log/pytorch_gpu1_fp32_bs128.txt
/imagenet/log/pytorch_gpu1_fp32_bs256.txt
/imagenet/log/pytorch_gpu1_amp_bs128.txt
/imagenet/log/pytorch_gpu1_amp_bs256.txt
/imagenet/log/pytorch_gpu8_fp32_bs128.txt
/imagenet/log/pytorch_gpu8_fp32_bs256.txt
/imagenet/log/pytorch_gpu8_amp_bs128.txt
/imagenet/log/pytorch_gpu8_amp_bs256.txt
```

在NGC报告的[Training performance benchmark](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5#training-performance-benchmark)小节，提供了其测试的参数配置。因此，我们提供的`test_all.sh`是参考了其文档中的配置。

> TODO(wanghuancoder):<br>
> 脚本路径

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

|卡数 | FP32(BS=128) | FP32(BS=max_144) | AMP(BS=128) | AMP(BS=max_256)|
|-----|-----|-----|-----|-----|
|1 | 355.69 | 368.89 | 780.48 | 797.38|
|8 | - | - | - | -|
|32 | - | - | - | -|

> TODO(wanghuancoder):<br>
> 完成测试，将1卡、8卡数据填入表格

> TODO(Distribute):<br>
> 完成测试，将32卡数据填入表格

## 五、日志数据
- [1卡 FP32 BS=128 日志](./logs/pytorch_gpu1_fp32_bs128.txt)
- [1卡 FP32 BS=256 日志](./logs/pytorch_gpu1_fp32_bs256.txt)
- [1卡 AMP BS=128 日志](./logs/pytorch_gpu1_amp_bs128.txt)
- [1卡 AMP BS=256 日志](./logs/pytorch_gpu1_amp_bs256.txt)

> TODO(wanghuancoder):<br>
> 完成测试，将1卡、8卡 与 公布性能数据 一致的原始日志文件提交到log目录下，并更新链接

> TODO(Distribute):<br>
> 完成测试，将32卡 与 公布性能数据 一致的原始日志文件提交到log目录下，并更新链接
