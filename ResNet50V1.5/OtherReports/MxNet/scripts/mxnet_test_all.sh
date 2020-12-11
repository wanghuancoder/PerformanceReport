#/bin/bash

python benchmark.py -n 1,8 -b 128 --dtype float32 -o benchmark_report_fp32.json -i 500 -e 3 -w 1 --num-examples 32000 --mode train > /data/imagenet/train-val-recordio-passthrough/log/mxnet_gpu1_gpu8_fp32_bs128.txt 2>&1

python benchmark.py -n 1,8 -b 256 --dtype float32 -o benchmark_report_fp32.json -i 500 -e 3 -w 1 --num-examples 32000 --mode train > /data/imagenet/train-val-recordio-passthrough/log/mxnet_gpu1_gpu8_fp32_bs256.txt 2>&1

python benchmark.py -n 1,8 -b 128 --dtype float16 -o benchmark_report_fp16.json -i 500 -e 3 -w 1 --num-examples 32000 --mode train > /data/imagenet/train-val-recordio-passthrough/log/mxnet_gpu1_gpu8_amp_bs128.txt 2>&1

python benchmark.py -n 1,8 -b 256 --dtype float16 -o benchmark_report_fp16.json -i 500 -e 3 -w 1 --num-examples 32000 --mode train > /data/imagenet/train-val-recordio-passthrough/log/mxnet_gpu1_gpu8_amp_bs256.txt 2>&1
