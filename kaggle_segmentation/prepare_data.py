import os
from select import select
import shutil
import cv2
import numpy as np
import random

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
        if cv2.imread(img_path, cv2.IMREAD_UNCHANGED) is not None:
            img_name_list = img_path.split('/')
            img_name = img_name_list[-3] + '_slice_'+ img_name_list[-1].split('_')[1] + '.png'
            shutil.copy(img_path, os.path.join(target_root_path,'label',img_name))
        else:
            print('Error loading: ', img_path)
       

def find_non_empty_data(mask_path):
    mask_list = os.listdir(mask_path)
    non_empty_list = []
    for mask_name in mask_list:
        msk = cv2.imread(os.path.join(mask_path, mask_name), cv2.IMREAD_UNCHANGED)
        if len(np.unique(msk)) > 1:
            non_empty_list.append(mask_name.split('.')[0])
    return non_empty_list


def convert_mask(src_mask_path, target_mask_path):
    all_mask_list = os.listdir(src_mask_path)
    for mask_name in all_mask_list:
        mask = cv2.imread(os.path.join(src_mask_path, mask_name), cv2.IMREAD_UNCHANGED)
        mask[mask==255] = 1
        cv2.imwrite(os.path.join(target_mask_path,mask_name),mask)
        
        

def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def compute_bce_class_weight(mask_path, num_class):
    mask_name_list = os.listdir(mask_path)
    pos_total = np.zeros(num_class)
    neg_total = np.zeros(num_class)
    for mask_name in mask_name_list:
        mask = cv2.imread(os.path.join(mask_path, mask_name), cv2.IMREAD_UNCHANGED)
        for i in range(num_class):
            msk = mask[:,:,i]
            pos = msk.sum()
            neg_msk = (1 - msk)
            neg = neg_msk.sum()
            pos_total[i] += pos
            neg_total[i] += neg
    return neg_total/pos_total



if __name__ == '__main__':
    mask_path = '/home/zhangzr/mmsegmentation_kaggle/data/kaggle_segmentation_data/label_3channel_convert'
    non_empty_list = find_non_empty_data(mask_path)
    with open('/home/zhangzr/mmsegmentation_kaggle/data/kaggle_segmentation_data/splits/non_empty.txt', 'w') as f:
        for item in non_empty_list:
            f.write(item + '\n')

    
    


    
    

   