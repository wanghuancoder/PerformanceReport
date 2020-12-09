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
|1 | 141.73 | 153.94 | 452.53 | 537.82|
|8 | - | - | - | -|

### 2.多机（32卡）测试

## 五、日志数据

> TODO: 待以 50个step 的日志替换

- [单卡 bs=32、FP32](./logs/tf_bert_pretraining_lamb_base_fp32_bs32_gpu1_gbs65536.log)
```
DLL 2020-12-08 10:21:53.954734 - Iteration: 41  throughput_train : 142.395 seq/s mlm_loss : 10.3715  nsp_loss : 0.6796  total_loss : 11.0512  avg_loss_step : 11.0362  learning_rate : 1.4625e-05
LL 2020-12-08 10:29:34.337979 - Iteration: 42  throughput_train : 142.487 seq/s mlm_loss : 10.3679  nsp_loss : 0.6650  total_loss : 11.0329  avg_loss_step : 11.0335  learning_rate : 1.50000005e-05
DLL 2020-12-08 10:37:15.950480 - Iteration: 43  throughput_train : 142.111 seq/s mlm_loss : 10.3302  nsp_loss : 0.6776  total_loss : 11.0079  avg_loss_step : 11.0296  learning_rate : 1.5375e-05
DLL 2020-12-08 10:45:00.087771 - Iteration: 44  throughput_train : 141.334 seq/s mlm_loss : 10.3645  nsp_loss : 0.7469  total_loss : 11.1114  avg_loss_step : 11.0246  learning_rate : 1.575e-05
DLL 2020-12-08 10:52:45.431648 - Iteration: 45  throughput_train : 140.962 seq/s mlm_loss : 10.3146  nsp_loss : 0.7337  total_loss : 11.0483  avg_loss_step : 11.0207  learning_rate : 1.6125001e-05
D
```

- [单卡 bs=64、FP32](./logs/tf_bert_pretraining_lamb_base_fp32_bs64_gpu1_gbs65536.log)
```
DLL 2020-12-07 16:38:37.360796 - Iteration: 41  throughput_train : 153.663 seq/s mlm_loss : 10.3616  nsp_loss : 0.6942  total_loss : 11.0558  avg_loss_step : 11.0463  learning_rate : 1.4625e-05
DLL 2020-12-07 16:45:43.281829 - Iteration: 42  throughput_train : 153.950 seq/s mlm_loss : 10.3603  nsp_loss : 0.6873  total_loss : 11.0477  avg_loss_step : 11.0419  learning_rate : 1.50000005e-05
DLL 2020-12-07 16:52:48.807443 - Iteration: 43  throughput_train : 154.092 seq/s mlm_loss : 10.3343  nsp_loss : 0.6519  total_loss : 10.9862  avg_loss_step : 11.0371  learning_rate : 1.5375e-05
DLL 2020-12-07 16:59:54.246930 - Iteration: 44  throughput_train : 154.124 seq/s mlm_loss : 10.3755  nsp_loss : 0.6641  total_loss : 11.0396  avg_loss_step : 11.0336  learning_rate : 1.575e-05
DLL 2020-12-07 17:06:59.364339 - Iteration: 45  throughput_train : 154.241 seq/s mlm_loss : 10.2902  nsp_loss : 0.6710  total_loss : 10.9612  avg_loss_step : 11.0277  learning_rate : 1.6125001e-05
```

- [单卡 bs=64、FP16](./logs/tf_bert_pretraining_lamb_base_fp16_bs64_gpu1_gbs65536.log)
```
DLL 2020-12-08 12:54:09.755333 - Iteration: 41  throughput_train : 453.548 seq/s mlm_loss : 10.3376  nsp_loss : 0.6920  total_loss : 11.0297  avg_loss_step : 11.0279  learning_rate : 1.4625e-05  loss_scaler : 134217728
DLL 2020-12-08 12:56:34.528833 - Iteration: 42  throughput_train : 453.388 seq/s mlm_loss : 10.3295  nsp_loss : 0.6840  total_loss : 11.0135  avg_loss_step : 11.0239  learning_rate : 1.50000005e-05  loss_scaler : 134217728
DLL 2020-12-08 12:58:59.693916 - Iteration: 43  throughput_train : 452.167 seq/s mlm_loss : 10.3364  nsp_loss : 0.6892  total_loss : 11.0256  avg_loss_step : 11.0183  learning_rate : 1.5375e-05  loss_scaler : 134217728
DLL 2020-12-08 13:01:25.008703 - Iteration: 44  throughput_train : 451.705 seq/s mlm_loss : 10.3236  nsp_loss : 0.6897  total_loss : 11.0132  avg_loss_step : 11.0110  learning_rate : 1.575e-05  loss_scaler : 134217728
DLL 2020-12-08 13:03:50.142865 - Iteration: 45  throughput_train : 452.266 seq/s mlm_loss : 10.2949  nsp_loss : 0.7061  total_loss : 11.0010  avg_loss_step : 11.0066  learning_rate : 1.6125001e-05  loss_scaler : 134217728
```

- [单卡 bs=128、FP16](./logs/tf_bert_pretraining_lamb_base_fp16_bs128_gpu1_gbs65536.log)

```
DLL 2020-12-08 04:58:55.265015 - Iteration: 41  throughput_train : 538.028 seq/s mlm_loss : 10.4172  nsp_loss : 0.6917  total_loss : 11.1089  avg_loss_step : 11.0981  learning_rate : 1.4625e-05  loss_scaler : 67108864
DLL 2020-12-08 05:00:57.222608 - Iteration: 42  throughput_train : 537.831 seq/s mlm_loss : 10.4127  nsp_loss : 0.6824  total_loss : 11.0951  avg_loss_step : 11.0940  learning_rate : 1.50000005e-05  loss_scaler : 67108864
DLL 2020-12-08 05:02:59.210241 - Iteration: 43  throughput_train : 537.698 seq/s mlm_loss : 10.3987  nsp_loss : 0.6798  total_loss : 11.0785  avg_loss_step : 11.0889  learning_rate : 1.5375e-05  loss_scaler : 67108864
DLL 2020-12-08 05:05:01.346357 - Iteration: 44  throughput_train : 537.047 seq/s mlm_loss : 10.3882  nsp_loss : 0.6856  total_loss : 11.0739  avg_loss_step : 11.0839  learning_rate : 1.575e-05  loss_scaler : 67108864
DLL 2020-12-08 05:07:03.450836 - Iteration: 45  throughput_train : 537.184 seq/s mlm_loss : 10.3898  nsp_loss : 0.6858  total_loss : 11.0756  avg_loss_step : 11.0795  learning_rate : 1.6125001e-05  loss_scaler : 67108864
```
