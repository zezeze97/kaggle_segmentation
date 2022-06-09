#!/bin/bash
export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=/home/zhangzr/mmsegmentation_kaggle:$PYTHONPATH
GPU=$2


# config=upernet_originsize_convnext_base_fp16_256x256_160k_kaggle_25d_multilabel
# config=upernet_originsize_convnext_base_fp16_320x384_160k_kaggle_25d_multilabel
# config=upernet_originsize_convnext_base_fp16_320x384_160k_kaggle_25d_multilabel_randomcrop
# config=upernet_originsize_convnext_base_fp16_320x384_160k_kaggle_25d_multilabel_mosaic
# config=upernet_originsize_convnext_base_fp16_320x384_16k_kaggle_25d_multilabel_mosaic_resize
#config=upernet_originsize_convnext_base_fp16_512x512_160k_kaggle_25d_multilabel
# config=upernet_originsize_convnext_base_fp16_320x384_160k_kaggle_25d_multilabel_diceloss
config=upernet_originsize_convnext_large_fp16_320x384_160k_kaggle_25d_multilabel
if [ $1 = "train" ]; then
    CUDA_VISIBLE_DEVICES=$GPU PORT=23475 ./tools/dist_train.sh configs/convnext/${config}.py 1 --work-dir cache/${config} 
    #CUDA_VISIBLE_DEVICES=$GPU PORT=23473 ./tools/dist_train.sh configs/swin/${config}.py 1 --work-dir cache/${config} 
elif [ $1 = "test" ]; then
    
    # CUDA_VISIBLE_DEVICES=$GPU python ./tools/test.py configs/convnext/${config}.py ../../input/mmsegckpts/iter_1600.pth --format-only --eval-options "imgfile_prefix=./test_results/upernet_convnext_base_fp16_256x256_16k_kaggle_no_crop"
    CUDA_VISIBLE_DEVICES=$GPU ./tools/dist_test.sh configs/convnext/${config}.py ./cache/upernet_originsize_convnext_base_fp16_320x384_160k_kaggle_25d_multilabel/best_mDice_iter_64000.pth 2 --eval mDice # --format-only --eval-options "imgfile_prefix=./test_results/upernet_convnext_base_fp16_256x256_16k_kaggle_no_crop"
fi

