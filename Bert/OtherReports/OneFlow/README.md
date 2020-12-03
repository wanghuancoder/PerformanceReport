# OneFlow Bert 性能复现

这里给出 OneFlow Bert Base Pre-Training 任务的性能测试报告。
对于FP32的性能测试，本报告严格按OneFlow公开的测试报告进行复现，对其提供的代码、脚本未做改动。其公开的测试报告请见：[《BERT base Benchmark Test Report》](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/637bb9cdb4cc1582f13bcc171acbc8a8089d9435/reports/bert_fp32_report.md)

对于 AMP 的性能测试，OneFlow 提供了 `--use_fp16` [配置选项](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/LanguageModeling/BERT/config.py#L57)，我们参考[OneFlow Benchmark Test Scripts](https://github.com/Oneflow-Inc/DLPerf/tree/master/OneFlow/scripts)指定了相关配置，完成了测试。

> TODO(Distribute):<br>
> 1. 需要确认OneFlow是否有公开的AMP的测试数据或者报告
> 2. 若以上表达有不符合实际测试的地方，需要分布式同学做修正。

## 目录
- [一、环境介绍](#一环境介绍)
- [二、环境搭建](#二环境搭建)
    * [1.单机（单卡、8卡）环境搭建](#1单机单卡8卡环境搭建)
    * [2.多机（32卡）环境搭建](#2多机32卡环境搭建)
- [三、测试步骤](#三测试步骤)
- [四、测试结果](#四测试结果)
- [五、日志数据](#五日志数据)
    * [1. 固定 BatchSize](#1-固定-batchsize)
    * [2. BatchSize 取 max](#2-batchsize-取-max)

## 一、环境介绍
环境介绍（物理机环境及Docker环境）在[《Paddle Bert Base 性能测试》](../../README.md#1物理机环境)中已经给出。

所有测试物理机环境完全一致，由于 OneFlow 没有提供Docker镜像，因此我们自建 Docker 进行了环境搭建。

## 二、环境搭建

>  TODO(Distribute)<br>
> 1. 提供 oneflow 使用的代码仓库链接<br>
> 2. 提供 oneflow 官方公布的环境搭建教程<br>
> 3. 下述侧重详细介绍 从用户角度复现 oneflow 的搭建流程，应包含执行命令、以及踩坑经验和解决方法（若有）

### 1.单机（单卡、8卡）环境搭建

> TODO(Distribute):<br>
> 1. 给出环境搭建详细文档 <br>
> 2. 包含 镜像准备、框架包安装、依赖库、模型代码、数据准备、执行script <br>
> 3. 如有踩坑地方，请额外说明

- 安装docker
```
docker pull xxxx
```

- 启动docker
```
nvidia-docker ...
```

- 下载数据

### 2.多机（32卡）环境搭建

> TODO(Distribute):<br>
> 1. 给出环境搭建详细文档<br>
> 2. 建议给出多机环境依赖库（如NCCL）安装教程、版本号、配置命令，注意事项等说明。


## 三、测试步骤

### 1.单机（单卡、8卡）测试

> TODO(Distribute):<br>
> 给出测试脚本，如果OneFlow提供了sh，必须使用OneFlow提供的sh
> FP32的sh在这个报告中有： <br>
> https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/reports/resnet50_v15_fp32_report.md <br>
> AMP的最好基于OneFlow的local_run.sh修改得到。

### 2.多机（32卡）测试

> TODO(Distribute):<br>
> 与单机要求一样

## 四、测试结果

- 训练吞吐率(sentences/sec)如下:

|卡数 | FP32(BS=32) | AMP(BS=64) | FP32(BS=max) | AMP(BS=max)|
|-----|-----|-----|-----|-----|
|1 | - | - | - | -|
|8 | - | - | - | -|
|32 | - | - | - | -|

> TODO(Distribute):<br>
> 完成测试，将数据填入表格
> 注意：FP32下BS为32， AMP下BS=64，最大BatchSize需要测一下来确定

## 五、日志数据

### 1. 固定 BatchSize

- [1卡 FP32 BS=32 日志](./logs/)
- [1卡 AMP BS=64 日志](./logs/)
- [8卡 FP BS=32 日志](./logs/)
- [8卡 AMP BS=64 日志](./logs/)
- [32卡 FP BS=32 日志](./logs/)
- [32卡 AMP BS=64 日志](./logs/)

### 2. BatchSize 取 max

- [1卡 FP32 BS=max 日志](./logs/)
- [1卡 AMP BS=max 日志](./logs/)
- [8卡 FP BS=max 日志](./logs/)
- [8卡 AMP BS=max 日志](./logs/)
- [32卡 FP BS=max 日志](./logs/)
- [32卡 AMP BS=max 日志](./logs/)

> TODO(Distribute):<br>
> 完成测试，原始日志文件提交到log目录下，并更新链接
