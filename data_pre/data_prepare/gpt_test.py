import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# 设置图片大小
img_size = (256, 256)

# 设置条纹的宽度 相当于像素之间的角度的变化速率  越大 条纹越细
stripe_width = 20

# 设置色轮的角速度 色轮一秒60圈
angular_velocity = 2 * np.pi / 60

# 计算每个像素所对应的时间间隔 相机的一秒30帧
time_per_pixel = 1 / 30

# 生成一个空白的图片
img = np.zeros((img_size[0], img_size[1], 3))

# 计算每个像素对应的角度
x, y = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
angle = y * stripe_width * angular_velocity * time_per_pixel

# 生成RGB颜色，通过调节振幅可以实现淡化效果
amplitude = 1
red = (amplitude * np.sin(angle + 0 * np.pi / 3) + 1) / 2
green = (amplitude * np.sin(angle + 2 * np.pi / 3) + 1) / 2
blue = (amplitude * np.sin(angle + 4 * np.pi / 3) + 1) / 2

# 将RGB颜色加入到图片中 plt是RGB , CV2是BGR
# img[:, :, 0] = red
# img[:, :, 1] = green
# img[:, :, 2] = blue

img[:, :, 2] = red * 255
img[:, :, 1] = green * 255
img[:, :, 0] = blue * 255
# 显示生成的图片
# plt.imshow(img)
# plt.axis('off')
# plt.show()

# srcImg = plt.imread('G:\Image_Decomposition\RainbowNet_Data\data_pre\data_prepare\picture\Places365_test_00000418.jpg')/255
srcImg = cv.imread('G:\Image_Decomposition\RainbowNet_Data\data_pre\data_prepare\picture\Places365_test_00000418.jpg')

# plt.imsave('G:/Image_Decomposition/RainbowNet_Data/data_pre/final_stripes.jpg', img)
cv.imwrite('G:/Image_Decomposition/RainbowNet_Data/data_pre/final_stripes.jpg', img)

# final_stripes = plt.imread('G:/Image_Decomposition/RainbowNet_Data/data_pre/final_stripes.jpg') / 255
final_stripes = cv.imread('G:/Image_Decomposition/RainbowNet_Data/data_pre/final_stripes.jpg')

# alpha = 0.85
# blended_img = alpha * srcImg + (1 - alpha) * final_stripes
blended_img = cv.addWeighted(srcImg, 0.80, final_stripes, 0.20, 0)

plt.imshow(blended_img)
plt.show()
plt.imsave('G:/Image_Decomposition/RainbowNet_Data/data_pre/final.jpg', blended_img)

