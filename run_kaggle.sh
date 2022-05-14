#!/bin/bash
export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=/home/zhangzr/mmsegmentation_kaggle:$PYTHONPATH
GPU=$2

config=upernet_convnext_base_fp16_512x512_160k_kaggle



if [ $1 = "train" ]; then
   
    CUDA_VISIBLE_DEVICES=$GPU PORT=23472 ./tools/dist_train.sh configs/convnext/${config}.py 1 --work-dir cache/${config} 
elif [ $1 = "test" ]; then
    
    CUDA_VISIBLE_DEVICES=$GPU python ./tools/test.py configs/convnext/${config}.py ./cache/${config}/iter_32000.pth --format-only --eval-options "imgfile_prefix=./test_results/upernet_convnext_base_fp16_512x512_160k_kaggle"
fi

