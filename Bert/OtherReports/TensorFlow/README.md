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

> TODO: 待以 50个step 的日志替换

|卡数 | FP32(BS=32) | FP32(BS=64) | AMP(BS=64) | AMP(BS=128)|
|-----|-----|-----|-----|-----|
|1 | 145.16 | 153.94 | 452.91 | 538.29|
|8 | - | - | - | -|

### 2.多机（32卡）测试

## 五、日志数据

> TODO: 待以 50个step 的日志替换

- [单卡 bs=32、FP32](./logs/tf_bert_pretraining_lamb_base_fp32_bs32_gpu1_gbs65536.log)
```
DLL 2020-12-07 20:04:34.092791 - Iteration: 16  throughput_train : 145.221 seq/s mlm_loss : 10.4521  nsp_loss : 0.6735  total_loss : 11.1256  avg_loss_step : 11.1381  learning_rate : 5.25e-06
DLL 2020-12-07 20:12:05.701908 - Iteration: 17  throughput_train : 145.250 seq/s mlm_loss : 10.4404  nsp_loss : 0.6923  total_loss : 11.1327  avg_loss_step : 11.1358  learning_rate : 5.625e-06
DLL 2020-12-07 20:19:37.382923 - Iteration: 18  throughput_train : 145.226 seq/s mlm_loss : 10.4118  nsp_loss : 0.6705  total_loss : 11.0823  avg_loss_step : 11.1347  learning_rate : 6e-06
DLL 2020-12-07 20:27:10.659157 - Iteration: 19  throughput_train : 145.141 seq/s mlm_loss : 10.4407  nsp_loss : 0.6946  total_loss : 11.1353  avg_loss_step : 11.1318  learning_rate : 6.3750003e-06
```

- [单卡 bs=64、FP32](./logs/tf_bert_pretraining_lamb_base_fp32_bs64_gpu1_gbs65536.log)
```
DLL 2020-12-07 16:52:48.807443 - Iteration: 43  throughput_train : 154.092 seq/s mlm_loss : 10.3343  nsp_loss : 0.6519  total_loss : 10.9862  avg_loss_step : 11.0371  learning_rate : 1.5375e-05
DLL 2020-12-07 16:59:54.246930 - Iteration: 44  throughput_train : 154.124 seq/s mlm_loss : 10.3755  nsp_loss : 0.6641  total_loss : 11.0396  avg_loss_step : 11.0336  learning_rate : 1.575e-05
DLL 2020-12-07 17:06:59.364339 - Iteration: 45  throughput_train : 154.241 seq/s mlm_loss : 10.2902  nsp_loss : 0.6710  total_loss : 10.9612  avg_loss_step : 11.0277  learning_rate : 1.6125001e-05
DLL 2020-12-07 17:14:05.275813 - Iteration: 46  throughput_train : 154.217 seq/s mlm_loss : 10.3395  nsp_loss : 0.6534  total_loss : 10.9929  avg_loss_step : 11.0250  learning_rate : 1.65e-05
DLL 2020-12-07 17:14:10.530985 -  throughput_train : 153.458 seq/s
```

- [单卡 bs=64、FP16](./logs/tf_bert_pretraining_lamb_base_fp16_bs64_gpu1_gbs65536.log)
```
DLL 2020-12-07 21:20:15.213285 - Iteration: 16  throughput_train : 452.515 seq/s mlm_loss : 10.4781  nsp_loss : 0.6814  total_loss : 11.1595  avg_loss_step : 11.1490  learning_rate : 5.25e-06  loss_scaler : 134217728
DLL 2020-12-07 21:22:40.396882 - Iteration: 17  throughput_train : 452.126 seq/s mlm_loss : 10.4795  nsp_loss : 0.6687  total_loss : 11.1482  avg_loss_step : 11.1474  learning_rate : 5.625e-06  loss_scaler : 134217728
DLL 2020-12-07 21:25:05.065196 - Iteration: 18  throughput_train : 453.729 seq/s mlm_loss : 10.4715  nsp_loss : 0.6940  total_loss : 11.1655  avg_loss_step : 11.1458  learning_rate : 6e-06  loss_scaler : 134217728
DLL 2020-12-07 21:27:30.365843 - Iteration: 19  throughput_train : 453.858 seq/s mlm_loss : 10.4725  nsp_loss : 0.6862  total_loss : 11.1588  avg_loss_step : 11.1437  learning_rate : 6.3750003e-06  loss_scaler : 134217728
```

- [单卡 bs=128、FP16](./logs/tf_bert_pretraining_lamb_base_fp16_bs128_gpu1_gbs65536.log)

```
DLL 2020-12-07 18:02:01.711027 - Iteration: 16  throughput_train : 539.905 seq/s mlm_loss : 10.4360  nsp_loss : 0.7037  total_loss : 11.1397  avg_loss_step : 11.1296  learning_rate : 5.25e-06  loss_scaler : 67108864
DLL 2020-12-07 18:04:03.470654 - Iteration: 17  throughput_train : 538.696 seq/s mlm_loss : 10.4206  nsp_loss : 0.6961  total_loss : 11.1167  avg_loss_step : 11.1263  learning_rate : 5.625e-06  loss_scaler : 67108864
DLL 2020-12-07 18:06:05.425436 - Iteration: 18  throughput_train : 537.847 seq/s mlm_loss : 10.4038  nsp_loss : 0.7048  total_loss : 11.1086  avg_loss_step : 11.1243  learning_rate : 6e-06  loss_scaler : 67108864
DLL 2020-12-07 18:08:07.661224 - Iteration: 19  throughput_train : 538.043 seq/s mlm_loss : 10.3948  nsp_loss : 0.6946  total_loss : 11.0894  avg_loss_step : 11.1200  learning_rate : 6.3750003e-06  loss_scaler : 67108864
```
