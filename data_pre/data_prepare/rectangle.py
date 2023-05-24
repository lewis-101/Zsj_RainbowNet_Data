import cv2
import numpy as np
import os


input_mash_path = 'G:/Image_Decomposition/Grounded-Segment-Anything-outputs/train/mask_out'
mash_path = 'G:/Image_Decomposition/Grounded-Segment-Anything-outputs/train/mask_out/763.png'
image_path = 'G:/Image_Decomposition/Grounded-Segment-Anything-outputs/train/image/763.png'
# for root, dirs, img_list in os.walk(input_mash_path):
#     print(img_list)
# 0读取灰度 1读取彩色
mask = cv2.imread(mash_path, 0)
# 查找非零值的位置
nz = np.nonzero(mask)

xmin = np.min(nz[1])
xmax = np.max(nz[1])
ymin = np.min(nz[0])
ymax = np.max(nz[0])
# 使用boundingRect函数计算mask的最小边界框 x是竖的方向 y是横的方向
x, y, w, h = cv2.boundingRect(np.column_stack(nz))

# 在原始图像上绘制矩形框
img = cv2.imread(image_path)
cv2.rectangle(img, (y, x), (y+h, x+w), (0, 255, 0), 2)

# 裁剪图像，获取矩形框内的像素
cropped_image = img[x:x+w, y:y+h]

# 可选：将裁剪后的图像缩放到固定的大小
# cropped_image = cv2.resize(cropped_image, (224, 224))



# 显示结果图像
cv2.imshow('image', img)
cv2.imshow('cropped_image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # 计算x和y的最小和最大值
# xmin = np.min(nz[1])
# xmax = np.max(nz[1])
# ymin = np.min(nz[0])
# ymax = np.max(nz[0])
#
# print(f"x轴的最小值：{xmin}")
# print(f"x轴的最大值：{xmax}")
# print(f"y轴的最小值：{ymin}")
# print(f"y轴的最大值：{ymax}")

# cv2.imshow('mask', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# mask_path =
# cv2.imread(mask_path)