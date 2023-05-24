import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def create_ranbinbow_stripes(srcImg):
    # 读入图像
    # img = cv.imread('G:\Image Decomposition\RainbowNet\data_prepare\picture\Places365_test_00000003.jpg')
    # 获取图像的高度和宽度
    height, width = srcImg.shape[:2]

    # 设置图片大小
    img_size = (height, width)

    # 设置条纹的宽度 相当于像素之间的角度的变化速率
    stripe_width = 25

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

    # 将RGB颜色加入到图片中
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

    # plt.imsave('G:/Image_Decomposition/RainbowNet_Data/data_pre/data_prepare/stripes/final_stripes.jpg', img)
    cv.imwrite('G:/Image_Decomposition/RainbowNet_Data/data_pre/data_prepare/stripes/final_stripes.jpg', img)

    return img
# plt 方式用 cv2 不用
def blend(srcImg, final_stripes, alpha):
    srcImg = srcImg / 255
    final_stripes = final_stripes / 255

    # alpha = 0.85
    blended_img = alpha * srcImg + (1 - alpha) * final_stripes

    # plt.imshow(blended_img)
    # plt.show()
    # plt.imsave('G:/Image Decomposition/RainbowNet_Data/data_pre/final.jpg', blended_img)
    return blended_img, srcImg

if __name__ == "__main__":
    path = 'picture/'
    num = 1
    first = 1
    alpha = 0.85
    for root, dirs, img_list in os.walk(path):
        print(img_list)
    for files in img_list:
        print(files)
        # srcImg = plt.imread('picture/' + files)
        srcImg = cv.imread('picture/' + files)
        srcImg = cv.resize(srcImg, (400, 400))

        # 灰度变RGB
        if len(srcImg.shape) == 2:
            srcImg = np.stack((srcImg,) * 3, axis=-1)

        if first == 1:
            # 创建条纹
            final_stripes = create_ranbinbow_stripes(srcImg)

        # 读取条纹
        # final_stripes = plt.imread('G:/Image_Decomposition/RainbowNet_Data/data_pre/data_prepare/stripes/final_stripes.jpg')
        final_stripes = cv.imread('G:/Image_Decomposition/RainbowNet_Data/data_pre/data_prepare/stripes/final_stripes.jpg')
        # 融合图像
        # blended_img, source_Img = blend(srcImg, final_stripes, alpha)
        blended_img = cv.addWeighted(srcImg, 0.80, final_stripes, 0.20, 0)
        # 保存
        # plt.imsave('../dataset/data/train/' + str(num) + '.jpg', blended_img)
        # plt.imsave('../dataset/gt/train/' + str(num) + '.jpg', source_Img)
        cv.imwrite('../dataset/data/train/' + str(num) + '.jpg', blended_img)
        cv.imwrite('../dataset/gt/train/' + str(num) + '.jpg', srcImg)

        first = first + 1
        num = num + 1












# if math.sin(0) <= sin_val <= math.sin(2 * math.pi / 3):
    #     return colors[0]
    # if math.sin(2 * math.pi / 3) < sin_val < math.sin(4 * math.pi / 3):
    #     return colors[1]
    # if math.sin(4 * math.pi / 3) <= sin_val <= math.sin(2 * math.pi) :
    #     return colors[2]

    # 红 (0,2 * math.pi / 3) 值(0,1)
    # if math.sin(0) <= sin_val <= 1.0:
    #     return colors[0]
    # # 绿 (2 * math.pi / 3，4 * math.pi / 3) 值(0.86,-0.86)
    # if math.sin(2 * math.pi / 3) < sin_val < math.sin(4 * math.pi / 3):
    #     return colors[1]
    # # 蓝(4 * math.pi / 3，0) 值(-0.86, 0)
    # if -1.0 <= sin_val <= math.sin(2 * math.pi):
    #     return colors[2]

    # # 红 (0, 2 * math.pi / 3) 值(0, 1)
    # if 0 <= rad < 2 * math.pi / 3:
    #     return colors[0]
    # # 绿 (2 * math.pi / 3, 4 * math.pi / 3) 值(0.86, -0.86)
    # if 2 * math.pi / 3 <= rad < 4 * math.pi / 3:
    #     return colors[1]
    # # 蓝(4 * math.pi / 3, 0) 值(-0.86, 0) (-1, 0)
    # if 4 * math.pi / 3 <= rad <= 2 * math.pi:
    #     return colors[2]

