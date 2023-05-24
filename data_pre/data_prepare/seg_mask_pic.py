import cv2
import numpy as np
import os

def create_ranbinbow_stripes(srcImg, stripes_path):
    # 读入图像
    # img = cv.imread('G:\Image_Decomposition\RainbowNet\data_prepare\picture\Places365_test_00000003.jpg')
    # 获取图像的高度和宽度
    height, width = srcImg.shape[:2]

    # 设置图片大小
    img_size = (height, width)

    # 设置条纹的宽度 相当于像素之间的角度的变化速率
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
    # stripes_path = 'G:/Image_Decomposition/RainbowNet_Data/data_pre/data_prepare/stripes/final_stripes.jpg'
    stripes_path = stripes_path
    # plt.imsave('G:/Image_Decomposition/RainbowNet_Data/data_pre/data_prepare/stripes/final_stripes.jpg', img)
    cv2.imwrite(stripes_path, img)

    return img, stripes_path


def seg_mask(img,mask):
    # 原图
    Original = img
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # ============要提高原图分辨率===========
    # # 定义目标分辨率
    # target_size = (2325, 1743)
    # # 使用双线性插值法提高分辨率
    # img= cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    # ====================================
    height, width = img.shape[0:2]
    mask = cv2.resize(mask, (width, height))

    # 设置填充参数
    top = bottom = 70
    left = right = 70
    # 对图像进行填充
    mask_t = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)  # 740 580
    mask_2 = mask_t
    # 定义一个 5x5 的椭圆形核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # 膨胀两次
    dilation1 = cv2.dilate(mask_2, kernel, iterations=2)
    dilation2 = cv2.dilate(dilation1, kernel2, iterations=2)
    # 腐蚀两次
    erosion2 = cv2.erode(dilation2, kernel2, iterations=2)
    mask_2 = cv2.erode(erosion2, kernel, iterations=2)
    # 去填充
    mask_2 = mask_2[top:-bottom, left:-right]  # 640 480

    mask_3 = mask_2 - mask  # 640 480 - 640 480
    # cv2.imshow('mask_3', mask_3)
    mask_4 = mask - mask_2  # 640 480 - 640 480
    # cv2.imshow('mask_4', mask_4)

    # cv2.imshow('mask', mask)

    # 前景mask
    _, binary_mask = cv2.threshold(mask_2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 背景mask
    bg_mask = cv2.bitwise_not(binary_mask)

    # 将掩模应用到原图像上，提取出分割目标的区域 前景和背景
    Segmented = cv2.bitwise_and(img, img, mask=binary_mask)
    Background = cv2.bitwise_and(img, img, mask=bg_mask)

    return Original, Segmented, Background

if __name__ == "__main__":
    input_path = 'G:/Image_Decomposition/Grounded-Segment-Anything-outputs/outputs4' # outputs
    output_path = 'G:/Image_Decomposition/Grounded-Segment-Anything-outputs/mask_pic4' # mask_pic
    raw_seg_out_path = 'G:/Image_Decomposition/Grounded-Segment-Anything-outputs/raw_seg_out4' # raw_seg_out
    Original_name = "Original.png"
    Segmented_name = "Segmented.png"
    Background_name = "Background.png"
    # ====================变成分割之后再加彩红条纹=========================
    # stripes_pat = 'G:/Image_Decomposition/RainbowNet_Data/data_pre/data_prepare/stripes/final_stripes.jpg'
    # first = 1
    # =============================================
    for root, dirs, files_list in os.walk(input_path):
        print(dirs)
        for files in dirs:# 每个子目录
            print(files)
            output_dir = os.path.join(output_path, files)
            raw_seg_out_name = files + ".png"
            # make dir
            os.makedirs(output_dir, exist_ok=True)

            root_dir = os.path.join(root, files)
            mask_name = 'mask.jpg'
            raw_img = 'raw_image.jpg'
            mask_path = os.path.join(root_dir, mask_name)
            img_path = os.path.join(root_dir, raw_img)
            mask = cv2.imread(mask_path)
            img = cv2.imread(img_path)

            # if files == "mask.jpg":
            #     mask_path = os.path.join(root, files)
            #     mask = cv2.imread(mask_path)
            # if files == "raw_image.jpg":
            #     img_path = os.path.join(root, files)
            #     img = cv2.imread(img_path)
            # os.path.join(output_dir, Original_name)
            Original, Segmented, Background = seg_mask(img, mask)
            # =============================变成分割之后再加彩红条纹============================
            # if len(Segmented.shape) == 2:
            #     Segmented = np.stack((Segmented,) * 3, axis=-1)
            #
            # if first == 1:
            #     # 创建条纹
            #     final_stripes, stripes_path = create_ranbinbow_stripes(Segmented, stripes_path)
            # # 读取条纹
            # final_stripes = cv2.imread(stripes_path)
            # # 融合图像 扭曲完以后再加条纹条纹
            # Segmented = cv2.addWeighted(Segmented, 0.85, final_stripes, 0.15, 0)
            # first = first + 1
            #=============================================================================
            cv2.imwrite(os.path.join(output_dir, Original_name), Original)
            cv2.imwrite(os.path.join(output_dir, Segmented_name), Segmented)
            cv2.imwrite(os.path.join(output_dir, Background_name), Background)
            # seg image_000001_1
            cv2.imwrite(os.path.join(raw_seg_out_path, raw_seg_out_name), Segmented)


