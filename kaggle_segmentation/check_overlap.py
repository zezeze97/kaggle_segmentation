import os
import cv2
import numpy as np

def check_overlap(mask_path):
    mask_list = os.listdir(mask_path)
    overlap_list = []
    for mask_name in mask_list:
        msk = cv2.imread(os.path.join(mask_path, mask_name), cv2.IMREAD_UNCHANGED)
        check_msk = np.sum(msk, axis=2)
        if np.any(check_msk>1):
            print(mask_name, 'exit overlap!!')
            overlap_list.append(mask_name)
    return overlap_list




if __name__ == '__main__':
    mask_path = '/home/zhangzr/mmsegmentation_kaggle/data/kaggle_segmentation_data/label_3channel_convert'
    overlap_list = check_overlap(mask_path)
    print(overlap_list)