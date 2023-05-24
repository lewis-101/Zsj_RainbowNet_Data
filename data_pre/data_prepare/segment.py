import cv2
import numpy as np

# 2325 1743
# 读入原图像和分割掩模
# img = cv2.imread('C://Users//11038//Desktop//pic//screen1.jpg')
# mask = cv2.imread('G://Image_Decomposition//segment-anything-main//result//output//screen1//44.png', 0)

# img = cv2.imread('G://Image_Decomposition//Grounded-Segment-Anything//outputs//image_000200_1//raw_image.jpg')
# mask = cv2.imread('G://Image_Decomposition//Grounded-Segment-Anything//outputs//image_000200_1//mask.jpg')
# img = cv2.imread('G:/Image_Decomposition/Grounded-Segment-Anything-outputs/outputs/image_000052_1//raw_image.jpg')
# mask = cv2.imread('G:/Image_Decomposition/Grounded-Segment-Anything-outputs/outputs/image_000052_1//mask.jpg')

# img = cv2.imread('G:/Image_Decomposition/Grounded-Segment-Anything-outputs/outputs/image_000451_1//raw_image.jpg')
# mask = cv2.imread('G:/Image_Decomposition/Grounded-Segment-Anything-outputs/outputs/image_000451_1//mask.jpg')

img = cv2.imread('G:/Image_Decomposition/NeuralMarker-main/data/flyingmarkers/test/images//image_000029_1.png')
mask = cv2.imread('G:/Image_Decomposition/NeuralMarker-main/data/flyingmarkers/test/mask_for_blend//image_000029.png')

mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# 释放窗口
print(len(mask))
print(len(mask[0]))
print(mask[0][0])

# cv2.imshow('mask', mask)

# ============要提高原图分辨率===========
# # 定义目标分辨率
# target_size = (2325, 1743)
# # 使用双线性插值法提高分辨率
# img= cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
# ====================================

height, width = img.shape[0:2]

mask = cv2.resize(mask, (width, height)) # 640 480

# 设置填充参数
top = bottom = 100
left = right = 100
# 对图像进行填充
mask_t = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0) # 740 580
# 显示填充后的图像
cv2.imshow('padded image', mask)
# mask = cv2.resize(mask, (width, height))
# mask = padder.unpad(mask)
# mask = mask[top:-bottom, left:-right]

mask_2 = mask_t  # 740 580
# 定义一个 5x5 的椭圆形核
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
# # 使用开运算去除小的噪点
# mask_2  = cv2.morphologyEx(mask_2 , cv2.MORPH_OPEN, kernel)
#
# # 使用闭运算填充大的空洞
# mask_2  = cv2.morphologyEx(mask_2 , cv2.MORPH_CLOSE, kernel)

# 膨胀两次
dilation1 = cv2.dilate(mask_2, kernel, iterations=2)
# cv2.imshow('mask', mask)
dilation2 = cv2.dilate(dilation1, kernel2, iterations=2)

# dilation3 = cv2.dilate(dilation2, kernel3, iterations=2)

# 腐蚀两次
# erosion1 = cv2.erode(dilation3, kernel3, iterations=2)
erosion2 = cv2.erode(dilation2, kernel2, iterations=2)
mask_2 = cv2.erode(erosion2, kernel, iterations=2)
# cv2.imshow('mask', mask)
mask_2 = mask_2[top:-bottom, left:-right] # 640 480

mask_3 = mask_2 - mask # 640 480 - 640 480
cv2.imshow('mask_3', mask_3)

mask_4 = mask - mask_2 # 640 480 - 640 480
cv2.imshow('mask_4', mask_4)

# 前景mask
_, binary_mask = cv2.threshold(mask_2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 背景mask
bg_mask = cv2.bitwise_not(binary_mask)
cv2.imshow('mask_2', mask_2)
cv2.imshow('mask', mask)

# 将掩模应用到原图像上，提取出分割目标的区域
img_seg = cv2.bitwise_and(img, img, mask=binary_mask)
img_seg_bg = cv2.bitwise_and(img, img, mask=bg_mask)




# 显示原图像和分割结果
# cv2.imshow('Original', img)
# cv2.imshow('Segmented', img_seg)
cv2.imwrite('G://Image_Decomposition//RainbowNet_Data//data_pre//data_prepare//segment_pic//Original.jpg', img)
cv2.imwrite('G://Image_Decomposition//RainbowNet_Data//data_pre//data_prepare//segment_pic//Segmented.jpg', img_seg)
cv2.imwrite('G://Image_Decomposition//RainbowNet_Data//data_pre//data_prepare//segment_pic//Background.jpg', img_seg_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()
