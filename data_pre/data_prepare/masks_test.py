import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde

# 加载所有掩码图像
mask_dir = 'G://Image_Decomposition//segment-anything-main//result//output//classroom//'
mask_files = os.listdir(mask_dir)

# 遍历所有掩码图像并检查像素值
# for mask_file in mask_files:
#     mask_path = os.path.join(mask_dir, mask_file)
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     unique_values = np.unique(mask)
#     if len(unique_values) == 1:
#         print('The pixel values in mask {} are all the same: {}'.format(mask_file, unique_values[0]))
#     else:
#         print('The pixel values in mask {} are not all the same.'.format(mask_file))

# # 遍历所有掩码图像并计算像素值分布
# mask_values = []
# for mask_file in mask_files:
#     if mask_file.endswith('.csv'):
#         continue
#     mask_path = os.path.join(mask_dir, mask_file)
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     mask_values.extend(mask[mask > 0].tolist()) # 只保留前景像素的值
#
# # 计算概率密度函数
# kde = gaussian_kde(mask_values)
#
# # 绘制概率密度函数
# x_grid = np.linspace(0, 255, 256)
# plt.plot(x_grid, kde(x_grid))
# plt.xlim([0, 255])
# plt.xlabel('Pixel Value')
# plt.ylabel('Density')
# plt.title('Mask Pixel Value Distribution')
# plt.show()

# mask_means = []
# mask_stds = []
# for mask_file in mask_files:
#     if mask_file.endswith('.csv'):
#         continue
#     mask_path = os.path.join(mask_dir, mask_file)
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     mask_values = mask[mask > 0].tolist() # 只保留前景像素的值
#     mask_mean = np.mean(mask_values)
#     mask_std = np.std(mask_values)
#     mask_means.append(mask_mean)
#     mask_stds.append
#
# # 根据均差阈值区分投影屏幕的掩码和其他物体的掩码
# mean_threshold = 56  # 均差阈值
# screen_mask_index = np.argmin(mask_means) # 找到均差最小的掩码的索引
# print(screen_mask_index)
# if mask_means[screen_mask_index] < mean_threshold:
#     screen_mask_path = mask_files[screen_mask_index]
#     print(f"投影屏幕掩码：{screen_mask_path}")
# else:
#     print("找不到投影屏幕掩码")
