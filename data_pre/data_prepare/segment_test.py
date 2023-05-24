import cv2
import numpy as np
import os
import torchvision.transforms as transforms

path = 'G://Image_Decomposition//segment-anything-main//result//output//classroom//'
# 读入原图像和分割掩模

img = cv2.imread('C://Users//11038//Desktop//pic//classroom.jpg')
# mask = cv2.imread('G://Image_Decomposition//segment-anything-main//result//output//classroom//44.png', 0)
# 最大的mask
max_sum = -1
# 最大的mask的路径
mask = path
for root, dirs, img_list in os.walk(path):
    print(img_list)
for files in img_list:
    # 不循环csv文件
    if files.endswith('.csv'):
        break
    print(files)
    sum_one = 0
    print(path + files)
    tmp_mask = cv2.imread(path + files, 0)
    tmp_mask[tmp_mask > 0] = 1 # 255 变成 1
    # 找到有最多 1 的mask" cpu 太慢
    # for i in len(tmp_mask):
    #     for j in len(tmp_mask[0]):
    #         if tmp_mask[i][j] > 0:
    #             sum_one = sum_one + 1

    # 使用 flat 属性遍历数组 cpu
    # for elem in tmp_mask.flat:
    #     if elem > 0:
    #              sum_one = sum_one + 1

    tmp_mask_tensor = transforms.ToTensor()(tmp_mask)

    # 将 PyTorch 张量移动到 GPU 上
    tmp_mask_tensor = tmp_mask_tensor.cuda()

    # 统计像素值大于 0 的个数
    sum_one = (tmp_mask_tensor > 0).sum().item()
    print("判断")
    # 判断个数
    if(sum_one > max_sum):
        max_sum = sum_one
        mask = tmp_mask
print(mask[0])
# 将掩模中的非目标区域设为 0，目标区域设为 1
mask[mask > 0] = 1

# 将掩模应用到原图像上，提取出分割目标的区域
img_seg = cv2.bitwise_and(img, img, mask=mask)

# 显示原图像和分割结果
# cv2.imshow('Original', img)
# cv2.imshow('Segmented', img_seg)
cv2.imwrite('G://Image_Decomposition//RainbowNet_Data//data_pre//data_prepare//segment_pic//Original.jpg', img)
cv2.imwrite('G://Image_Decomposition//RainbowNet_Data//data_pre//data_prepare//segment_pic//Segmented.jpg', img_seg)
cv2.waitKey(0)
cv2.destroyAllWindows()
