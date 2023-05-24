import cv2 as cv
import numpy as np
import os
import math
# path = 'picture/'
path = 'G:/Image_Decomposition/RainbowNet_Data/data_pre/data_prepare/picture/' # 绝对路径
train_path = 'G:/Image_Decomposition/RainbowNet_Data/data_pre/dataset/data/train/'
gt_path = 'G:/Image_Decomposition/RainbowNet_Data/data_pre/dataset/gt/train/'
num = 1
def create_ranbinbow(srcImg):
    # 读入图像
    # img = cv.imread('G:\Image Decomposition\RainbowNet\data_prepare\picture\Places365_test_00000003.jpg')


    # 获取图像的高度和宽度
    # height, width = img.shape[:2]
    height, width = srcImg.shape[:2]

    # 创建一个与原图像相同大小的图像，用于绘制条纹
    stripes = np.zeros((height, width, 3), dtype=np.uint8)

    # 设置条纹的颜色顺序（红、绿、蓝）和宽度
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    stripe_height = 10

    # 设置周期函数的参数
    amplitude = 128
    frequency = 2 * math.pi / width

    # 画横向的条纹 要与角速度的周期函数结合
    for i in range(0, height, stripe_height * len(colors)):
        for j, color in enumerate(colors):
            x1, y1 = 0, i + j * stripe_height
            x2, y2 = width, y1 + stripe_height
            cv.rectangle(stripes, (x1, y1), (x2, y2), color, -1)

    # 将条纹与原图像相加，并保存结果图像
    # final_Img = cv.addWeighted(img, 0.95, stripes, 0.05, 0)
    final_Img = cv.addWeighted(srcImg, 0.95, stripes, 0.05, 0)

    # cv.imwrite('output_image.jpg', final_Img)

    # return final_Img, source_Img
    return final_Img, srcImg

if __name__ == "__main__":
    for root, dirs, img_list in os.walk(path):
        print(img_list)
    for files in img_list:
        print(files)
        # srcImg = cv.imread('picture/' + files)
        srcImg = cv.imread(path + files)
        srcImg = cv.resize(srcImg, (400, 400))
        # dstImg, cutImg, source_Img = create_fish(srcImg, k1, k2, k3, k4)
        # 创建图像
        final_Img, source_Img = create_ranbinbow(srcImg)
        # output = cv.addWeighted(img, 0.95, stripes, 0.05, 0)
        # cv.imwrite('../dataset/data/train/' + str(num) + '.jpg', final_Img)
        # cv.imwrite('../dataset/gt/train/' + str(num) + '.jpg', source_Img)
        cv.imwrite(train_path + str(num) + '.jpg', final_Img)
        cv.imwrite(gt_path + str(num) + '.jpg', source_Img)
        #cv.imwrite('../dataset/data/train/' + str(num) + '.jpg', output)
        #cv.imwrite('../dataset/gt/train/' + str(num) + '.jpg', source_Img)

        num = num + 1

