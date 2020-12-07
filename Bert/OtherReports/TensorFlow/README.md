<!-- omit in toc -->
# NGC TensorFlow Bert 性能复现

此处给出了基于 [NGC TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) 实现的 Bert Base Pre-Training 任务的详细复现流程，包括执行环境、Paddle版本、环境搭建、复现脚本、测试结果和测试日志。

<!-- omit in toc -->
## 目录
- [一、环境介绍](#一环境介绍)
  - [1.物理机环境](#1物理机环境)
  - [2.Docker 镜像](#2docker-镜像)
- [二、环境搭建](#二环境搭建)
  - [1. 拉取代码](#1-拉取代码)
  - [2. 构建镜像](#2-构建镜像)
  - [3. 准备数据](#3-准备数据)
- [三、测试步骤](#三测试步骤)
- [四、测试结果](#四测试结果)
  - [1.单机（单卡、8卡）测试](#1单机单卡8卡测试)
  - [2.多机（32卡）测试](#2多机32卡测试)
- [五、日志数据](#五日志数据)


## 一、环境介绍

### 1.物理机环境

我们使用了同一个物理机环境，对 [NGC TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) 的 Bert 模型进行了测试，详细物理机配置，见[Paddle Bert Base 性能测试](../../README.md#1.物理机环境)。

### 2.Docker 镜像

NGC TensorFlow 的代码仓库提供了自动构建 Docker 镜像的的 [shell 脚本](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/scripts/docker/build.sh)，

- 镜像版本：`nvcr.io/nvidia/tensorflow:20.06-tf1-py3`

## 二、环境搭建

我们遵循了 NGC TensorFlow 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT#quick-start-guide) 教程成功搭建了测试环境，主要过程如下：
### 1. 拉取代码

```bash
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/TensorFlow/LanguageModeling/BERT
```

### 2. 构建镜像
```bash
bash scripts/docker/build.sh   # 构建镜像
bash scripts/docker/launch.sh  # 启动容器
```

我们将 `launch.sh` 脚本中的 `docker` 命令换为了 `nvidia-docker` 启动的支持 GPU 的容器，其他均保持不变，脚本如下：
```bash
#!/bin/bash

CMD=${@:-/bin/bash}
NV_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"all"}

nvidia-docker run --name=test_tf_bert -it \
    --net=host \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NVIDIA_VISIBLE_DEVICES=$NV_VISIBLE_DEVICES \
    -v $PWD:/workspace/bert \
    -v $PWD/results:/results \
    bert $CMD
```

### 3. 准备数据

NGC TensorFlow 提供单独的数据下载和预处理脚本 [data/create_datasets_from_start.sh](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/data/create_datasets_from_start.sh)。在容器中执行如下命令，可以下载和制作 `wikicorpus_en` 的 tfrecord 数据集。

```bash
bash data/create_datasets_from_start.sh wiki_only
```

> TODO(Aurelius84): 确定是否要提供一份处理好的 wikipedia 的 tfrecord 样本数据集链接。

由于数据集比较大，且容易受网速的影响，上述命令执行时间较长。因此，为了更方便复现竞品的性能数据，我们提供了已经处理好的 tfrecord 格式[样本数据集]()。

## 三、测试步骤

为了更准确的复现 NGC TensorFlow 公布的 [NVIDIA DGX-1 (8x V100 32GB)](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT#pre-training-training-performance-single-node-on-dgx-1-32GB) 性能数据，我们严格按照官方提供的模型代码配置、启动脚本，进行了的性能测试。

官方提供的 [scripts/run_pretraining_lamb.sh](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/scripts/run_pretraining_lamb.sh) 执行脚本中，默认配置的是两阶段训练。我们此处统一仅执行 **第一阶段训练**，并根据日志中的输出的数据计算吞吐。

**重要的配置参数：**

- **train_batch_size_phase1**: 用于指定每张卡上的 batch_size 数目
- **precision**: 用于指定精度训练模式，fp32 或 fp16
- **use_xla**: 是否开启 XLA 加速，我们统一开启此选项
- **num_gpus**: 用于指定 GPU 卡数
- **bert_model**: 用于指定 Bert 模型，我们统一指定为 **base**

为了更方便地测试不同 batch_size、num_gpus、precision组合下的 Pre-Training 性能，我们单独编写了 `run_benchmark.sh` 脚本，并放在`scripts`目录下。

shell 脚本内容如下：
```bash
#!/bin/bash

set -x

batch_size=$1  # batch size per gpu
num_gpus=$2    # number of gpu
precision=$3   # fp32 | fp16
num_accumulation_steps_phase1=$(expr 65536 \/ $batch_size \/ $num_gpus)
train_steps=${4:-200}        # max train steps
bert_model=${5:-"base"}      # base | large

# run pre-training
bash scripts/run_pretraining_lamb.sh $batch_size 64 8 7.5e-4 5e-4 $precision true $num_gpus 2000 200 $train_steps 200 $num_accumulation_steps_phase1 512 $bert_model
```

## 四、测试结果
### 1.单机（单卡、8卡）测试

**单卡启动脚本：**

若测试单机单卡 batch_size=32、FP32 的训练性能，执行如下命令：

```bash
bash scripts/run_benchmark.sh 32 1 fp32
```

**8卡启动脚本：**

若测试单机8卡 batch_size=64、FP16 的训练性能，执行如下命令：

```bash
bash scripts/run_benchmark.sh 64 8 fp16
```

|卡数 | FP32(BS=32) | FP32(BS=64) | AMP(BS=64) | AMP(BS=128)|
|-----|-----|-----|-----|-----|
|1 | - | - | - | -|
|8 | - | - | - | -|

### 2.多机（32卡）测试

## 五、日志数据
