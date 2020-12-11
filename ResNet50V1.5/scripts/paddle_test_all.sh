export CUDA_VISIBLE_DEVICES=0

python ./tools/static/train.py -c ResNet50_1gpu_fp32_bs128.yaml > paddle_gpu1_fp32_bs128.txt

python ./tools/static/train.py -c ResNet50_1gpu_fp32_bs256.yaml > paddle_gpu1_fp32_bs256.txt

python ./tools/static/train.py -c ResNet50_1gpu_amp_bs128.yaml > paddle_gpu1_amp_bs128.txt

python ./tools/static/train.py -c ResNet50_1gpu_amp_bs256.yaml > paddle_gpu1_amp_bs256.txt

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" ./tools/static/train.py -c ResNet50_8gpu_fp32_bs128.yaml > paddle_gpu8_fp32_bs128.txt

python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" ./tools/static/train.py -c ResNet50_8gpu_fp32_bs256.yaml > paddle_gpu8_fp32_bs256.txt

python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" ./tools/static/train.py -c ResNet50_8gpu_amp_bs128.yaml > paddle_gpu8_amp_bs128.txt

python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" ./tools/static/train.py -c ResNet50_8gpu_amp_bs256.yaml > paddle_gpu8_amp_bs256.txt