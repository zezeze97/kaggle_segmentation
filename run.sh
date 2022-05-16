#!/bin/bash
export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=/home/zhangzr/mmsegmentation_kaggle:$PYTHONPATH
GPU=$2

#config=upernet_convnext_base_fp16_256x256_16k_kaggle
config=upernet_convnext_base_fp16_256x256_16k_kaggle_no_crop



if [ $1 = "train" ]; then
   
    CUDA_VISIBLE_DEVICES=$GPU PORT=23473 ./tools/dist_train.sh configs/convnext/${config}.py 2 --work-dir cache/${config} 
elif [ $1 = "test" ]; then
    
    # CUDA_VISIBLE_DEVICES=$GPU python ./tools/test.py configs/convnext/${config}.py ../../input/mmsegckpts/iter_1600.pth --format-only --eval-options "imgfile_prefix=./test_results/upernet_convnext_base_fp16_256x256_16k_kaggle_no_crop"
    CUDA_VISIBLE_DEVICES=$GPU ./tools/dist_test.sh configs/convnext/${config}.py ./cache/upernet_convnext_base_fp16_256x256_16k_kaggle_no_crop/iter_9600.pth 1 --format-only --eval-options "imgfile_prefix=./test_results/upernet_convnext_base_fp16_256x256_16k_kaggle_no_crop"
fi

