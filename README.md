# Kaggle 比赛

该仓库用于[UW-Madison GI Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/overview)比赛

# 数据集

[预处理数据集下载](https://disk.pku.edu.cn:443/link/EDDA036A4F65315386C15FB2793EBFB4)

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
pip install -e .  
```


# 训练，测试

```sh
# 训练
bash run.sh train $GPU
# 测试
bash run.sh test $GPU
```

# 在线可视化预测

[kaggle_segmentation/inference_demo.ipynb](kaggle_segmentation/inference_demo.ipynb)
