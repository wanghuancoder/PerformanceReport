# NGC TensorFlow ResNet50V1.5 性能测试

这里给出 NGC TensorFlow ResNet50V1.5的性能测试报告。
对于1卡、8卡性能测试，本报告严格按NGC公开的测试报告进行复现，对其提供的代码、脚本未做改动。其公开的测试报告请见：[《ResNet-50 v1.5 for TensorFlow》](https://github.com/Oneflow-Inc/DLPerf/tree/master/NVIDIADeepLearningExamples/TensorFlow/Classification/ConvNets/resnet50v1.5)

对于32卡性能测试，由于NGC并未提供测试环境和测试方法，我们参考[XXX](www.baidu.com)搭建了测试环境，完成了测试。
> TODO(Distribute):<br>
> 找到一个TF或NGC官方的、关于分布式使用的文档，放在XXX位置，并提供链接。后续的测试，也真正参考这个文档进行测试。

## 目录

## 一、环境介绍
环境介绍（物理机环境及Docker环境）在[《Paddle ResNet50V1.5 性能测试》](../../)中已经给出。
所有测试物理机环境完全一致，Docker环境使用NVIDIA官方提供的`NGC 20.03`镜像。
> TODO(wanghuancoder):<br>
> 确认一下最终是否使用NGC 20.03

## 二、环境搭建

### 1.单机（单卡、8卡）环境搭建

> TODO(wanghuancoder):<br>
> 1. 严格按以下脚本复现
> 2. TF AMP慢问题的分析

- 安装docker
```
docker pull xxxx
```

- 启动docker
```
nvidia-docker ...
```

- 下载数据

请参考如下脚本搭建环境：
```
# 1. 准备最新测试代码，进入测试代码所在目录
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/TensorFlow/Classification/ConvNets
# 2. 下载数据，并处理数据
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
# 3. 创建并执行docker
docker build . -t nvidia_rn50
sudo nvidia-docker run --rm -it -v /ssd2/wanghuan29/ILSVRC2012_tf_records/tf_records/train_val:/data/tfrecords --ipc=host nvidia_rn50
# 4. 生成dali数据
bash ./utils/dali_index.sh /data/tfrecords /data/tfrecords/dali_idx
# 5. 执行training_perf.sh进行测试
bash resnet50v1.5/training/training_perf.sh
```
执行测试脚本后，即可获得性能数据，如下图所示：

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

> TODO(wanghuancoder):<br>
> 1. 严格按以下脚本复现
> 2. TF AMP慢问题的分析

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

|卡数 | FP32(BS=128) | FP32(BS=128) | AMP(BS=128) | AMP(BS=256)|
|-----|-----|-----|-----|-----|
|1 | - | - | - | -|
|8 | - | - | - | -|
|32 | - | - | - | -|

> TODO(wanghuancoder):<br>
> 完成测试，将1卡、8卡数据填入表格

> TODO(Distribute):<br>
> 完成测试，将32卡数据填入表格

## 五、日志数据
- [1卡 FP32 BS=128 日志](./log/)
- ...

> TODO(wanghuancoder):<br>
> 完成测试，将1卡、8卡 与 公布性能数据 一致的原始日志文件提交到log目录下，并更新链接

> TODO(Distribute):<br>
> 完成测试，将32卡 与 公布性能数据 一致的原始日志文件提交到log目录下，并更新链接
