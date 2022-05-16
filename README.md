# Kaggle 比赛

该仓库用于[UW-Madison GI Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/overview)比赛

# 数据集探索
各类别占比
{3: 9996781, 2: 18230834, 1: 19962696, 0: 2854342809}

# 环境安装
```sh
conda create -n mmseg-kaggle python=3.10 -y
conda activate mmseg-kaggle
# conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install -e .  
```

# 数据集准备

将数据集下载到input文件夹下，并运行[kaggle_segmentation/prepare_data.ipynb](kaggle_segmentation/prepare_data.ipynb)

# 训练，测试

```sh
# 训练
bash run.sh train $GPU
# 测试
bash run.sh test $GPU
```

# 在线可视化预测
[kaggle_segmentation/inference_demo.ipynb](kaggle_segmentation/inference_demo.ipynb)