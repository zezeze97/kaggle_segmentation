import cv2
import numpy as np
from collections import Counter
import os

dir_path = './data/mmseg_train_25d_carno/labels'
img_paths = os.listdir(dir_path)


res = {}
for i in range(3):
    res[i] = {0:0,1:0}
for img_path in img_paths:
    img_path = os.path.join(dir_path, img_path)
    gt_semantic_seg = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    for k in range(3):
        cls_dict = np.bincount(gt_semantic_seg[:,:,k].flatten())
        res[k][0] += cls_dict[0]
        if len(cls_dict)==2:
            res[k][1] += cls_dict[1]
for k in range(3):
    total = res[k][0] + res[k][1]
    res[k][0] /= total
    res[k][1] /= total
    res[k]['ratio'] = res[k][0] / res[k][1]
print(res)



