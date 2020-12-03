# OneFlow ResNet50V1.5 性能测试

这里给出 OneFlow ResNet50V1.5的性能测试报告。
对于FP32的性能测试，本报告严格按OneFlow公开的测试报告进行复现，对其提供的代码、脚本未做改动。其公开的测试报告请见：[《OneFlow ResNet50-V1.5 Benchmark Test Report》](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/reports/resnet50_v15_fp32_report.md)

对于AMP的性能测试，由于OneFlow并未提供相关的测试报告，我们参考[OneFlow 官方Benchmark文档](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/Classification/cnns)搭建了测试环境，完成了测试。
> TODO(Distribute):<br>
> 1. OneFlow确实没有公开过AMP的测试报告吧？
> 2. 以上表达有不符合实际测试的地方，需要分布式同学做修正。

## 目录

## 一、环境介绍
环境介绍（物理机环境及Docker环境）在[《Paddle ResNet50V1.5 性能测试》](../../)中已经给出。

所有测试物理机环境完全一致，由于OneFlow没有提供Docker镜像，因此我们自建Docker进行了环境搭建。

## 二、环境搭建

### 1.单机（单卡、8卡）环境搭建

> TODO(Distribute):<br>
> 给出环境搭建详细文档

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
> 给出环境搭建详细文档


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

- 训练吞吐率(images/sec)如下:

|卡数 | FP32(BS=128) | FP32(BS=max_128) | AMP(BS=128) | AMP(BS=max_256)|
|-----|-----|-----|-----|-----|
|1 | - | - | - | -|
|8 | - | - | - | -|
|32 | - | - | - | -|

> TODO(Distribute):<br>
> 完成测试，将数据填入表格
> 注意：BS为128和最大BatchSize，最大BatchSize需要咱们测一下来确定

## 五、日志数据
- [1卡 FP32 BS=128 日志](./log/)
- ...

> TODO(Distribute):<br>
> 完成测试，原始日志文件提交到log目录下，并更新链接
