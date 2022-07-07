#!/bin/bash
export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=/home/zhangzr/mmsegmentation_kaggle:$PYTHONPATH
GPU=$2



# config=smp_unet_swin_base_patch4_window7_384x384_160k_kaggle25d_pretrain_224x224_22K
# config=upernet_swin_base_patch4_window7_384x384_160k_kaggle25d_pretrain_224x224_22K_TTA
# config=upernet_originsize_convnext_large_fp16_384x384_160k_kaggle_25d_multilabel_carno_rotate_tta
# config=smp_unet_swin_base_patch4_window7_384x384_160k_kaggle25d_pretrain_224x224_22K_TTA
# config=upernet_swin_base_patch4_window7_384x384_160k_kaggle25d_pretrain_224x224_22K_TTA_Dice
# config=upernet_swin_base_patch4_window7_384x384_160k_kaggle25d_pretrain_224x224_22K_TTA_class_weight
# config=upernet_swin_base_patch4_window7_384x384_160k_kaggle25d_pretrain_224x224_22K_TTA_OHEM
# config=upernet_swin_base_patch4_window7_384x384_160k+160k_kaggle25d_pretrain_224x224_22K_TTA_OHEM
# config=upernet_swinv2_base_160k_pretrained_patch4_window12_192_22k_OHEM_TTA
# config=upernet_swin_large_patch4_window12_384x384_320k_kaggle25d_pretrain_384x384_22K_TTA_OHEM
config=upernetori_swinv2_base_patch4_window12to24_192to384_22kto1k_OHEM_TTA
if [ $1 = "train" ]; then
    # CUDA_VISIBLE_DEVICES=$GPU PORT=23471 ./tools/dist_train.sh configs/convnext/${config}.py 2 --work-dir cache/${config} 
    # CUDA_VISIBLE_DEVICES=$GPU PORT=23472 ./tools/dist_train.sh configs/swin/${config}.py 2 --work-dir cache/${config} 
    CUDA_VISIBLE_DEVICES=$GPU PORT=23472 ./tools/dist_train.sh configs/swin_v2/${config}.py 2 --work-dir cache/${config} 
elif [ $1 = "test" ]; then
    
    # CUDA_VISIBLE_DEVICES=$GPU python ./tools/test.py configs/convnext/${config}.py ../../input/mmsegckpts/iter_1600.pth --format-only --eval-options "imgfile_prefix=./test_results/upernet_convnext_base_fp16_256x256_16k_kaggle_no_crop"
    CUDA_VISIBLE_DEVICES=$GPU ./tools/dist_test.sh configs/convnext/${config}.py ./cache/upernet_originsize_convnext_base_fp16_320x384_160k_kaggle_25d_multilabel/best_mDice_iter_64000.pth 2 --eval mDice # --format-only --eval-options "imgfile_prefix=./test_results/upernet_convnext_base_fp16_256x256_16k_kaggle_no_crop"
fi

