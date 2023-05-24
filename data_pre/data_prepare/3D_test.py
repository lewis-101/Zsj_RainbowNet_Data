import cv2 as cv
import numpy as np
import os
# 读取原始图片
# 有条纹的 G:\Image Decomposition\RainbowNet_Data\data_pre\dataset\data\train\1.jpg

# img = cv2.imread("G:/Image Decomposition/RainbowNet_Data/data_pre/data_prepare/picture/Places365_test_00000001.jpg")
# 保存路径
train_path = 'G:/Image_Decomposition/RainbowNet_Data/data_pre/dataset/3D_data/train/'
# 彩虹绝对路径
path = 'G:/Image_Decomposition/RainbowNet_Data/data_pre/dataset/data/train/'
num = 1

def create_wrap(srcRanImg):
    # 加载图像
    # image = cv.imread("G:\Image Decomposition\RainbowNet_Data\data_pre\dataset\data/train/1.jpg")

    # 定义拱形曲面的形状
    height, width = srcRanImg.shape[:2]
    R = 500  # 球面曲面半径
    center_x, center_y = width // 2, height // 2  # 曲面中心点
    # 生成坐标系
    x, y = np.meshgrid(np.arange(width) - center_x, np.arange(height) - center_y) # 二维矩阵
    # 球面曲面方程的解 1
    z = np.sqrt(R ** 2 - x ** 2 - y ** 2)

    # 定义曲面方程 2
    # a = 10  # 曲率系数
    # b = 20  # 控制曲面的陡峭程度
    # z = a * (np.tanh(b * np.sqrt(x ** 2 + y ** 2) / 1000) - np.tanh(
    #     b * np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / 1000))

    # 定义曲面方程 3
    # a = 10  # 控制曲面高度
    # b = 20  # 控制曲面陡峭程度
    # z = a * np.exp(-((x - srcRanImg.shape[1] // 2)**2 + (y - srcRanImg.shape[0] // 2)**2) / b)


    # 定义相机参数
    focal_length = 500
    camera_matrix = np.array([[focal_length, 0, center_x], [0, focal_length, center_y], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4,1))

    # 计算投影位置
    object_points = np.stack((x, y, z), axis=2)
    object_points = object_points.reshape(-1, 3)
    image_points, _ = cv.projectPoints(object_points, (0, 0, 0), (0, 0, 0), camera_matrix, dist_coeffs)
    image_points = image_points.reshape(height, width, 2)

    # 映射图像到曲面上
    final_Img = cv.remap(srcRanImg, image_points[..., 0].astype(np.float32), image_points[..., 1].astype(np.float32), cv.INTER_LINEAR)
    return final_Img, srcRanImg



if __name__ == "__main__":
    for root, dirs, img_list in os.walk(path):
        print(img_list)
    for files in img_list:
        print(files)
        # 加载图像
        srcRanImg = cv.imread(path + files)
        srcRanImg = cv.resize(srcRanImg, (400, 400))

        # 创建图像
        final_Img, source_Img = create_wrap(srcRanImg)
        # train图片
        cv.imwrite(train_path + str(num) + '.jpg', final_Img)
        # gt图片
        # cv.imwrite(gt_path + str(num) + '.jpg', source_Img)


        num = num + 1
    # 显示结果
    # cv.imwrite(train_path+'1.1.jpg',dst)
    # cv.imshow("Original Image", image)
    # cv.imshow("Projected Image", dst)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


