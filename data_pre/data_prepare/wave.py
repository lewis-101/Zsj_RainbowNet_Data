
import cv2
import numpy as np
import pywt

# 读取RGB图像并转换为BGR格式
img = cv2.imread('G:/Image_Decomposition/RainbowNet_Data/data_pre/final.jpg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# 将BGR图像拆分为三个通道
b, g, r = cv2.split(img)

# 对每个通道进行小波变换
wavelet = 'haar'
coeffs_r = pywt.dwt2(r, wavelet)
coeffs_g = pywt.dwt2(g, wavelet)
coeffs_b = pywt.dwt2(b, wavelet)

# 将每个通道的高频系数设置为零
coeffs_r = list(coeffs_r)
coeffs_g = list(coeffs_g)
coeffs_b = list(coeffs_b)
coeffs_r[1] = (coeffs_r[1][0], np.zeros_like(coeffs_r[1][1]), np.zeros_like(coeffs_r[1][2]))
coeffs_g[1] = (coeffs_g[1][0], np.zeros_like(coeffs_g[1][1]), np.zeros_like(coeffs_g[1][2]))
coeffs_b[1] = (coeffs_b[1][0], np.zeros_like(coeffs_b[1][1]), np.zeros_like(coeffs_b[1][2]))

# 对每个通道进行小波逆变换
r_restored = pywt.idwt2(coeffs_r, wavelet)
g_restored = pywt.idwt2(coeffs_g, wavelet)
b_restored = pywt.idwt2(coeffs_b, wavelet)


# 将3个通道合并为一张图像
restored = cv2.merge((r_restored, g_restored, b_restored))
# 将输出通道的顺序调整为RGB
restored = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)


# 显示原始图像和处理后的图像
cv2.imshow('Original', img)
cv2.imshow('Restored', restored)
cv2.waitKey(0)
cv2.destroyAllWindows()

