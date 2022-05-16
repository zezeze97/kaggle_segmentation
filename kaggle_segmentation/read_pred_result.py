import cv2
import os
import numpy as np

if __name__ == '__main__':
    pred_result_root_path = './test_results/upernet_convnext_base_fp16_256x256_160k_kaggle_no_crop'
    pred_img_name_list = os.listdir(pred_result_root_path)
    for pred_img_name in pred_img_name_list:
        pred_img = cv2.imread(os.path.join(pred_result_root_path, pred_img_name), cv2.IMREAD_UNCHANGED)
        if 3 in np.unique(pred_img):
            print(np.unique(pred_img))
            print(pred_img.shape)
            print(pred_img_name)
