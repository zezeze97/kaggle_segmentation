import os
from select import select
import shutil
import cv2
import numpy as np
import random
from tqdm import tqdm
# Sklearn
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
import pandas as pd

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
    for mask_name in tqdm(all_mask_list):
        msk =np.load(os.path.join(src_mask_path, mask_name)).astype('float32')
        msk/=255.0
        msk = msk.astype(np.uint8)
        save_path = os.path.join(target_mask_path, mask_name.split('.')[0] + '.png')
        cv2.imwrite(save_path, msk)

def convert_img(src_img_path, target_img_path):
    all_img_list = os.listdir(src_img_path)
    for img_name in tqdm(all_img_list):
        img = np.load(os.path.join(src_img_path, img_name))
        img = (img - img.min())/(img.max() - img.min())*255.0 # scale image to [0, 255]
        img = img.astype('uint8') # uint16 -> uint8
        save_path = os.path.join(target_img_path, img_name.split('.')[0] + '.png')
        cv2.imwrite(save_path, img)
        
        

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

def convert_split(src_split_list):
    convert_name_list = []
    for img_name in src_split_list:
        img_name_list = img_name.split('_')
        new_name = img_name_list[0] + '_' + img_name_list[1] + '_' + 'slice_' + img_name_list[2].zfill(5)
        convert_name_list.append(new_name)
    return convert_name_list



def split_data_new(df):
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups = df["case"])):
        df.loc[val_idx, 'fold'] = fold
    fold_0 = df[df['fold']==0]
    fold_1 = df[df['fold']==1]
    fold_2 = df[df['fold']==2]
    fold_3 = df[df['fold']==3]
    fold_4 = df[df['fold']==4]
    fold_0_list = [];fold_1_list = [];fold_2_list = [];fold_3_list = [];fold_4_list = []
    for index,row in fold_0.iterrows():
        fold_0_list.append(row['id'])
    for index,row in fold_1.iterrows():
        fold_1_list.append(row['id'])
    for index,row in fold_2.iterrows():
        fold_2_list.append(row['id'])
    for index,row in fold_3.iterrows():
        fold_3_list.append(row['id'])
    for index,row in fold_4.iterrows():
        fold_4_list.append(row['id'])


    return fold_0_list, fold_1_list, fold_2_list, fold_3_list, fold_4_list



if __name__ == '__main__':
    df = pd.read_csv('/home/zhangzr/mmsegmentation_kaggle/data/2_5d_seg_data/train.csv')
    fold_0_list, fold_1_list, fold_2_list, fold_3_list, fold_4_list = split_data_new(df)
    train_fold_0 = fold_1_list + fold_2_list + fold_3_list + fold_4_list
    val_fold_0 = fold_0_list
    train_fold_1 = fold_0_list + fold_2_list + fold_3_list + fold_4_list
    val_fold_1 = fold_1_list
    train_fold_2 = fold_0_list + fold_1_list + fold_3_list + fold_4_list
    val_fold_2 = fold_2_list
    train_fold_3 = fold_1_list + fold_2_list + fold_0_list + fold_4_list
    val_fold_3 = fold_3_list
    train_fold_4 = fold_1_list + fold_2_list + fold_3_list + fold_0_list
    val_fold_4 = fold_4_list
    with open('/home/zhangzr/mmsegmentation_kaggle/data/2_5d_seg_data/splits/train_fold_0.txt', 'w') as f:
        for item in train_fold_0:
            f.write(item + '\n')
    with open('/home/zhangzr/mmsegmentation_kaggle/data/2_5d_seg_data/splits/val_fold_0.txt', 'w') as f:
        for item in val_fold_0:
            f.write(item + '\n')


    


    
    


    
    

   