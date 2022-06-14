# Kaggle 比赛

该仓库用于[UW-Madison GI Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/overview)比赛



# 数据集探索

[参考](https://www.kaggle.com/code/andradaolteanu/aw-madison-eda-in-depth-mask-exploration)

# 环境安装

```sh
conda create -n mmseg-kaggle python=3.10 -y
conda activate mmseg-kaggle
# conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
git clone https://github.com/zezeze97/kaggle_segmentation.git
cd {path of project}
pip install -e .  
```
# 数据集下载，预处理

从官网下载好数据集后，放在该项目的input目录下，运行[kaggle_segmentation/prepare_data.ipynb](kaggle_segmentation/prepare_data.ipynb)


# 训练，测试

```sh
# 训练
bash run.sh train $GPU
# 测试
bash run.sh test $GPU
```

# 可视化预测

[kaggle_segmentation/inference_demo.ipynb](kaggle_segmentation/inference_demo.ipynb)

# TODO

- 图片尺寸较小, 可以尝试upernet origin size?(默认1/4大小) 
- train pipline去除random crop
- 图片case的相关性，更好的建模方式？
- 2.5d data？
- 更换backbone, swin transformer
- mutilabel问题，最后激活使用sigmoid而不是softmax!!
- 实验结果整理
- EfficientNet + Unet?  (他山之石)
