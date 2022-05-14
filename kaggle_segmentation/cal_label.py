import cv2
import numpy as np
from collections import Counter
import os

dir_path = './mmseg_train_data/labels'
# img_paths = os.listdir(dir_path)
img_paths = []
with open('./mmseg_train_data/splits/fold_0.txt', 'r') as f:
    infos = f.readlines()
    for info in infos:
        img_paths.append(info.strip()+'.png')

res = {}
for i in range(4):
    res[i] = 0
for img in img_paths:
    print(img)
    img = os.path.join(dir_path, img)
    gt_semantic_seg = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    print(np.unique(gt_semantic_seg))
    # print(type(np.unique(img)[0]))
    cls_dict = Counter(gt_semantic_seg.ravel())
    for k in res.keys():
        if k in cls_dict.keys():
            res[k] += cls_dict[k]
print(res)
res = dict(sorted(res.items(), key=lambda d:d[1]))
print(res)



