import numpy as np
import cv2
import os

mean = []
std = []
img_list = []

# Train dataset
dir_path = './pre_data/pre_train_GF'
img_paths = os.listdir(dir_path)

count = 0
for img_path in img_paths:
    img_path = os.path.join(dir_path, img_path)
    count += 1
    print(img_path)
    # img = tff.imread(img_path)
    # img = img.transpose((1, 2, 0))
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = img[::, np.newaxis]
    img_list.append(img)
print(count)

# Test dataset
dir_path = './pre_data/pre_testA_GF'
img_paths = os.listdir(dir_path)

for img_path in img_paths:
    img_path = os.path.join(dir_path, img_path)
    count += 1
    print(img_path)
    # img = tff.imread(img_path)
    # img = img.transpose((1, 2, 0))
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = img[::, np.newaxis]
    img_list.append(img)
print(count)

# ==================

imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 1023.0  # 255.0

for i in range(3): # 4
    channel = imgs[:, :, i, :].ravel()
    mean.append(np.mean(channel))
    std.append(np.std(channel))

# mean.reverse()
# std.reverse()
print(count)
print(mean)
print(std)





