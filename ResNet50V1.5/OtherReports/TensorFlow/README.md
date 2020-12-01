# NGC TensorFlow ResNet50V1.5 性能复现报告

这里给出 NGC TensorFlow ResNet50V1.5的性能复现报告，本报告严格按NGC提供的报告进行复现。

## 执行环境

- GPU：NVIDIA V100-SXM2-16GB
- CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
- 内存：64 GB
- 系统：Ubuntu 18.04.4 LTS
- CUDA：11
- cuDNN：8.0.4

## Docker环境

本次复现使用NVIDIA官方提供的`NGC 20.03`镜像, 其软件配置如下：
- Ubuntu18.04
- Python 3.6
- TensorFlow 1.15.2
- CUDA 10.2.89
- cuDNN 7.6.5
- NCCL 2.6.3
- Horovod 0.19.0
- OpenMPI 3.1.4
- DALI 0.19.0

## 测试说明

测试过程严格按照[NGC官网](https://github.com/Oneflow-Inc/DLPerf/tree/master/NVIDIADeepLearningExamples/TensorFlow/Classification/ConvNets/resnet50v1.5)公开的报告执行，对齐提供的代码、脚本未做改动。具体测试内容说明如下：
- Node
- BatchSize
- FP32/AMP
- XLA

## 环境搭建&测试

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

## 测试结果

- 训练吞吐率(images/sec)如下:

|卡数 | FP32(BS=128) | FP32(BS=160) | AMP(BS=128) | AMP(BS=208)|
|-----|-----|-----|-----|-----|
|1 | - | - | - | -|
|8 | - | - | - | -|
|32 | - | - | - | -|

多卡加速比如下：

|卡数 | FP32(BS=128) | FP32(BS=160) | AMP(BS=128) | AMP(BS=208) |
|-----|-----|-----|-----|-----|
|1 | - | - | - | - |
|8 | - | - | - | - |
|32 | - | - | - | - |

# 日志数据