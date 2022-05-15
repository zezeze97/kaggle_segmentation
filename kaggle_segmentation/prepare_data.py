import os
import shutil
import cv2
import numpy as np


def list_allfile(path,all_files=[]):    
    if os.path.exists(path):
        files=os.listdir(path)
    else:
        print('this path not exist')
    for file in files:
        if os.path.isdir(os.path.join(path,file)):
            list_allfile(os.path.join(path,file),all_files)
        else:
            all_files.append(os.path.join(path,file))
    return all_files

def move(src_root_path, target_root_path):
    if not os.path.exists(target_root_path):
        os.mkdir(target_root_path)
        os.mkdir(os.path.join(target_root_path, 'label'))

    all_file_list = list_allfile(src_root_path)
    for img_path in all_file_list:
        img_name_list = img_path.split('/')
        img_name = img_name_list[-3] + '_slice_'+ img_name_list[-1].split('_')[1] + '.png'
        shutil.copy(img_path, os.path.join(target_root_path,'label',img_name))

def clean_data(src_img_root_path, src_label_root_path, target_root_path):
    all_file_list = list_allfile(src_img_root_path)
    for img_path in all_file_list:
        img_name = img_path.split('/')[-1]
        mask = cv2.imread(os.path.join(src_label_root_path, img_name), cv2.IMREAD_UNCHANGED)
        if len(np.unique(mask)) > 1:
            shutil.copy(os.path.join(src_label_root_path, img_name), os.path.join(target_root_path, 'label_3channel',img_name))
            shutil.copy(os.path.join(src_img_root_path, img_name), os.path.join(target_root_path, 'train', img_name))

def convert_mask(src_mask_path, target_mask_path):
    all_mask_list = os.listdir(src_mask_path)
    for mask_name in all_mask_list:
        mask = cv2.imread(os.path.join(src_mask_path, mask_name), cv2.IMREAD_UNCHANGED)
        mask[mask == 255] = 1
        argmax_mask = np.argmax(mask, axis=2) + 1
        background_mask = np.max(mask, axis=2).astype(np.int8) - 1
        new_mask = argmax_mask + background_mask
        print(np.unique(new_mask))
        cv2.imwrite(os.path.join(target_mask_path,mask_name),new_mask)

if __name__ == '__main__':
    src_mask_path = '/home/zhangzr/mmsegmentation_kaggle/data/kaggle_segmentation_clean_data/label_3channel'
    target_mask_path = '/home/zhangzr/mmsegmentation_kaggle/data/kaggle_segmentation_clean_data/label'
    convert_mask(src_mask_path, target_mask_path)