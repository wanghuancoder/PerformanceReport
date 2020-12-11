<!-- omit in toc -->
# Paddle Bert Base 性能测试

此处给出了基于 Paddle 框架实现的 Bert Base Pre-Training 任务的训练性能详细测试报告，包括执行环境、Paddle 版本、环境搭建、复现脚本、测试结果和测试日志。

相同环境下，其他深度学习框架的 Bert 训练性能数据测试流程，请参考：[OtherReports](./OtherReports)。

<!-- omit in toc -->
## 目录
- [一、测试说明](#一测试说明)
- [二、环境介绍](#二环境介绍)
  - [1.物理机环境](#1物理机环境)
  - [2.Docker 镜像](#2docker-镜像)
- [三、环境搭建](#三环境搭建)
  - [1.单机（单卡、8卡）环境搭建](#1单机单卡8卡环境搭建)
  - [2.多机（32卡）环境搭建](#2多机32卡环境搭建)
- [四、测试步骤](#四测试步骤)
  - [1.单机（单卡、8卡）测试](#1单机单卡8卡测试)
  - [2.多机（32卡）测试](#2多机32卡测试)
- [五、测试结果](#五测试结果)
  - [1.Paddle训练性能](#1paddle训练性能)
  - [2.与业内其它框架对比](#2与业内其它框架对比)
- [六、日志数据](#六日志数据)



## 一、测试说明

我们统一使用了 **吞吐能力** 作为衡量性能的数据指标。**吞吐能力** 是业界公认的、最主流的框架性能考核指标，它直接体现了框架训练的速度。

Bert Base 模型是自研语言处理领域极具代表性的模型，包括 Pre-Training 和 Fine-tune 两个子任务，此处我们选取 Pre-Training 阶段作为测试目标。在测试性能时，我们以 **sentences/sec）** 作为训练期间的吞吐性能。在其它框架中，默认也均采用相同的计算方式。

测试中，我们选择如下3个维度，测试吞吐性能：

- **卡数**

   本次测试关注1卡、8卡、32卡情况下，模型的训练吞吐性能。选择的物理机是单机8卡配置。
   因此，1卡、8卡测试在单机下完成。32卡在4台机器下完成。

- **FP32/AMP**

   FP32 和 AMP 是业界框架均支持的两种精度训练模式，也是衡量框架性能的混合精度量化训练的重要维度。
   本次测试分别对 FP32 和 AMP 两种精度模式进行了测试。


- **BatchSize**

   经调研，大多框架的 Bert Base Pre-Training 任务在第一阶段 max_seq_len=128的数据集训练时 ，均支持 FP32 模式下 BatchSize=32，AMP 模式下 BatchSize=64。因此我们分别测试了上述两种组合方式下的吞吐性能。

关于其它一些参数的说明：

- **XLA**

   本次测试的原则是测试 Bert Base 在 Paddle 下的最好性能表现，同时对比其与其它框架最好性能表现的优劣。

   因此，对于支持 XLA 的框架，我们默认打开 XLA 模式，已获得该框架最好的吞吐性能数据。

- **优化器**

   > TODO(Aurelius84): 最终确认 Paddle 使用的优化器类型
   在 Bert Base 的 Pre-Training 任务上，各个框架使用的优化器略有不同。NGC TensorFlow、NGC PyTorch、PaddlePaddle 均支持 LAMBOptimizer，OneFlow默认仅支持了AdamOptimizer。

   此处我们以各个框架默认使用的优化器为准，并测试模型的吞吐性。

## 二、环境介绍
### 1.物理机环境

- 系统：CentOS Linux release 7.5.1804
- GPU：Tesla V100-SXM2-32GB * 8
- CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz * 40
- CUDA：11
- cuDNN：8.0.4
- 内存：502 GB

### 2.Docker 镜像
> TODO(Aurelius84): 待更新Paddle开源出去的docker镜像tags

- 镜像版本：`paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82`
- Paddle 版本：`2.0rc1`
- CUDA 版本：`10`
- cuDnn 版本： `7.6.5`


## 三、环境搭建

各深度学习框架在公开的 Github 仓库中给出了详细的docker镜像和构建脚本，具体搭建流程请参考：[此处](./OtherReports)。

如下是 Paddle 测试环境的具体搭建流程:

### 1.单机（单卡、8卡）环境搭建

- **拉取代码**
  ```bash
  git clone https://github.com/PaddlePaddle/models.git
  cd models
  ```


- **构建镜像**

   ```bash
   # 拉取镜像
   docker pull hub.baidubce.com/paddlepaddle/paddle-benchmark:cuda10.1-cudnn7-runtime-ubuntu16.04

   # 创建并进入容器
   nvidia-docker run --name=test_bert_paddle -it \
    --net=host \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -v $PWD:/workspace/models \
    hub.baidubce.com/paddlepaddle/paddle-benchmark:cuda10.1-cudnn7-runtime-ubuntu16.04 /bin/bash
   ```

- **安装Paddle**
   ```bash
   # 安装paddle whl包 (todo: 待更新)
   pip3.7 install -U paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
   ```

- **准备数据**

   > TODO(Aurelius84): 待上传样本数据集，并给出下载链接和解压路径

   Bert 模型的 Pre-Training 任务是基于 [wikipedia]() 和 [BookCorpus]() 数据集进行的训练的，原始数据集比较大。我们提供了一份小的、且已处理好的[样本数据集]()，可以下载并解压到`models/bert_data`目录里。


### 2.多机（32卡）环境搭建

> TODO(Distribute):<br>
> 1. 提供分布式测试环境搭建的详细方法，可参考OneFlow的报告：<br>
> https://github.com/Oneflow-Inc/DLPerf/tree/master/PaddlePaddle/bert#nccl <br>
> https://github.com/Oneflow-Inc/DLPerf/tree/master/PaddlePaddle/bert#2%E6%9C%BA16%E5%8D%A1 <br>
> 2. 注意：咱们Paddle也计划制作Docker镜像，将必要的环境安装在镜像中，如果分布式的环境搭建可以预安装到Docker中，请分布式同学联系王欢，共同制作Docker。而能够在Docker中预安装好的环境，可以在文档的环境搭建介绍中不提供具体安装方法。

## 四、测试步骤

在 [benchmark/bert](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/benchmark/bert) 目录下，我们提供了分别用于单机测试的 `run_pretrain_single.py` 脚本和用于多机测试的 `run_pretrain.py` 脚本。

**重要参数：**
- **model_type**: 训练模型的类型，此处统一指定为 `bert`
- **model_name_or_path:** 预训练模型的名字或路径，此处统一指定为 `bert-base-uncased`
- **batch_size:** 每张 GPU 上的 batch_size 大小
- **use_amp:** 使用是否混合精度训练
- **enable_addto:** 是否开启梯度的 `addto` 聚合策略，默认开启
- **max_steps:** 设置训练的迭代次数，统一设置为5000
- **logging_steps:** 日志打印的步长，统一设置为100


### 1.单机（单卡、8卡）测试

为了更方便的复现我们的测试结果，我们提供一键测试 benchmark 数据的脚本 `run_benchmark.sh` ，需放在 `benchmark/bert`目录下。

- **脚本内容如下：**
   ```bash
   #!/bin/bash

   export PYTHONPATH=/workspace/models/PaddleNLP
   export DATA_DIR=/workspace/models/bert_data/
   export CUDA_VISIBLE_DEVICES=0

   batch_size=${1:-32}
   use_amp=${2:-"True"}
   max_steps=${3:-200}
   logging_steps=${4:-100}

   python3.7 ./run_pretrain_single.py \
      --model_type bert \
      --model_name_or_path bert-base-uncased \
      --max_predictions_per_seq 20 \
      --batch_size $batch_size   \
      --learning_rate 1e-4 \
      --weight_decay 1e-2 \
      --adam_epsilon 1e-6 \
      --warmup_steps 10000 \
      --input_dir $DATA_DIR \
      --output_dir ./tmp2/ \
      --logging_steps $logging_steps \
      --save_steps 50000 \
      --max_steps $max_steps \
      --use_amp $use_amp\
      --enable_addto True
   ````

> TODO: 给出单机单卡、8卡的执行命令


### 2.多机（32卡）测试
> TODO(分布式):(需包含)<br>
> 1. 提供多机32卡可修改配置的同一个执行shell脚本，给出脚本文件链接<br>
> 2. 对重要参数进行逐一说明<br>
> 3. 给出多机32卡的执行命令

## 五、测试结果

### 1.Paddle训练性能
- 训练吞吐率(sequences/sec)如下:

|卡数 | FP32(BS=32) | AMP(BS=64) | FP32(BS=64) | AMP(BS=128) |
|-----|-----|-----|-----|-----|
|1 | - | - | - | - |
|8 | - | - | - | - |
|32 | - | - | - | - |


### 2.与业内其它框架对比

说明：
- 同等执行环境下测试
- 单位：`sequences/sec`
- 对于支持 `DALI/XLA` 的框架，以下测试为开启 `DALI/XLA` 的数据
- BatchSize 选用各框架支持的最大 BatchSize

| 参数 | PaddlePaddle | NGC TensorFlow 1.15 | NGC PyTorch |
|-----|-----|-----|-----|
| FP32 GPU=1,BS=64 | - | 153.94 | 127.02 |
| AMP GPU=1,BS=128 | - | 538.29 | 527.38 |
| FP32 GPU=8,BS=64 | - | - | - |
| AMP GPU=8,BS=128 | - | - | - |
| FP32 GPU=32,BS=64 | - | - | - |
| AMP GPU=32,BS=128 | - | - | - |

## 六、日志数据

> TODO(Aurelius84):<br>
> 完成测试，将1卡、8卡 与 公布性能数据 一致的原始日志文件提交到log目录下，并更新链接

> TODO(Distribute):<br>
> 完成测试，将32卡 与 公布性能数据 一致的原始日志文件提交到log目录下，并更新链接
