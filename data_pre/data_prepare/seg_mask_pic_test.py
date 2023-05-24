import cv2
import numpy as np
import os


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
    # # 膨胀两次
    # dilation1 = cv2.dilate(mask_2, kernel, iterations=2)
    # dilation2 = cv2.dilate(dilation1, kernel2, iterations=2)
    # # 腐蚀两次
    # erosion2 = cv2.erode(dilation2, kernel2, iterations=2)
    # mask_2 = cv2.erode(erosion2, kernel, iterations=2)
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
    # input_path = 'G:/Image_Decomposition/Grounded-Segment-Anything-outputs/outputs4' # outputs
    input_path = 'G:/Image_Decomposition/NeuralMarker-main/data/flyingmarkers/validation_test/images_mask'  # outputs
    output_path = 'G:/Image_Decomposition/Grounded-Segment-Anything-outputs/outputs_image' # mask_pic_test
    seg_out_path = 'G:/Image_Decomposition/Grounded-Segment-Anything-outputs/train/image' # raw_seg_out
    mask_out_path = 'G:/Image_Decomposition/Grounded-Segment-Anything-outputs/train/mask_out'
    Original_name = "Original.png"
    Segmented_name = "Segmented.png"
    Background_name = "Background.png"
    number_name = 1
    for root, dirs, img_list in os.walk(input_path):
        print(img_list)
    for files in img_list:
        filesName = files.rsplit('.',1)[0]
        filename_last = filesName.rsplit('_', 1)[-1]
        last_char = filename_last
        #  图名 去找 对应的 mask
        if last_char == "1":
            print(filesName)
            output_dir_tmp = output_path + "/" + filesName[0:-1] + "1"
            # make dir
            os.makedirs(output_dir_tmp, exist_ok=True)
            # load image
            image_path = input_path + "/" + files

            mask_name = filesName[0:-2] + ".png"
            mask_path = os.path.join(input_path, mask_name)
            raw_img = files
            img_path = os.path.join(input_path, raw_img)

            mask = cv2.imread(mask_path)
            img = cv2.imread(img_path)
            # 最后保存图片的名字
            final_seg_image_name = files
            Original, Segmented, Background = seg_mask(img, mask)

            cv2.imwrite(os.path.join(output_dir_tmp, Original_name), Original)
            cv2.imwrite(os.path.join(output_dir_tmp, Segmented_name), Segmented)
            cv2.imwrite(os.path.join(output_dir_tmp, Background_name), Background)
            # seg image_000001_1
            # cv2.imwrite(os.path.join(seg_out_path, final_seg_image_name), Segmented)
            cv2.imwrite(os.path.join(seg_out_path, str(number_name) + ".png"), Segmented)
            # mask_out
            cv2.imwrite(os.path.join(mask_out_path, str(number_name) + ".png"), mask)
            number_name = number_name + 1
    # ===========================image_free==================================
    print("image_free")
    input_free_path = 'G:/Image_Decomposition/NeuralMarker-main/data/flyingmarkers/validation_test/images_free_mask'
    output_free_path = 'G:/Image_Decomposition/Grounded-Segment-Anything-outputs/outputs_free'
    seg_free_path = 'G:/Image_Decomposition/Grounded-Segment-Anything-outputs/train/image_free'
    number_free_name = 1
    for root, dirs, img_list in os.walk(input_free_path):
        print(img_list)
    for files in img_list:
        filesName = files.rsplit('.',1)[0]
        filename_last = filesName.rsplit('_', 1)[-1]
        last_char = filename_last
        #  图名 去找 对应的 mask
        if last_char == "1":
            print(filesName)
            output_dir_tmp = output_free_path + "/" + filesName[0:-1] + "1"
            # make dir
            os.makedirs(output_dir_tmp, exist_ok=True)
            # load image
            image_path = input_free_path + "/" + files

            mask_name = filesName[0:-2] + ".png"
            mask_path = os.path.join(input_free_path, mask_name)
            raw_img = files
            img_path = os.path.join(input_free_path, raw_img)

            mask = cv2.imread(mask_path)
            img = cv2.imread(img_path)
            # 最后保存图片的名字
            final_seg_free_name = files
            Original, Segmented, Background = seg_mask(img, mask)

            cv2.imwrite(os.path.join(output_dir_tmp, Original_name), Original)
            cv2.imwrite(os.path.join(output_dir_tmp, Segmented_name), Segmented)
            cv2.imwrite(os.path.join(output_dir_tmp, Background_name), Background)
            # seg image_000001_1 => 1.png
            # cv2.imwrite(os.path.join(seg_free_path, final_seg_free_name), Segmented)
            cv2.imwrite(os.path.join(seg_free_path, str(number_free_name) + ".png"), Segmented)
            number_free_name = number_free_name + 1