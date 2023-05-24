import math
import cv2 as cv
import numpy as np
import os


def get_color(angle):
    # 将角度值转换为弧度值     360 = 2 * pi
    rad = angle * math.pi / 180.0
    print("rad值:", rad)
    # 计算 sin 函数值
    sin_val = math.sin(rad)
    print("sin_val值:",sin_val)
    # 设置条纹的颜色顺序（红、绿、蓝）
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # 渐变系数
    Gradient = 1
    # 渐变范围
    Gra_range = math.pi / 12
    # 三分点
    axis_0 = 0.0
    axis_1 = 2 * math.pi / 3
    axis_2 = 4 * math.pi / 3
    axis_3 = 2 * math.pi
    # 根据 rad 值确定颜色  红、绿、蓝
    print("====================================")
    # 如果这个渐变不对的话 比如说[128,127,0] Gradient 和 (1.0 - Gradient) 我是不是应该让颜色清淡一些
    # 渐变 红不变蓝少 (0,15)
    if axis_0 <= rad < (axis_0 + Gra_range):
        print((axis_0 + Gra_range))
        Gradient = (axis_0 + Gra_range) - rad
        # Gradient = Gradient / Gra_range * 0.5
        Gradient = Gradient / Gra_range
        print(((axis_0 + Gra_range) - rad) % Gra_range)
        print("右Gradient值:", Gradient)
        # result = [Gradient * colors[2][i] + (1.0 - Gradient) * colors[0][i] for i in range(len(colors[0]))]
        result = [Gradient * colors[2][i] + colors[0][i] for i in range(len(colors[0]))]
        result = list(map(int, map(round, result)))
        return result
    # 标准红
    if (axis_0 + Gra_range) <= rad < (axis_1 - Gra_range):
        result = colors[0]
        return result
    # 渐变 红绿 (105,135)
    if (axis_1 - Gra_range) <= rad < (axis_1 + Gra_range):
        # 右边 红少绿不变 跟右边的值比
        if rad - axis_1 > 0:
            Gradient = (axis_1 + Gra_range) - rad
            # Gradient = Gradient / Gra_range * 0.5
            Gradient = Gradient / Gra_range
            print(((axis_1 + Gra_range) - rad) % Gra_range)
            print("右Gradient值:", Gradient)
            # result = [Gradient * colors[0][i] + (1.0 - Gradient) * colors[1][i] for i in range(len(colors[0]))]
            result = [Gradient * colors[0][i] + colors[1][i] for i in range(len(colors[0]))]
            result = list(map(int, map(round, result)))
            return result
        # 左边 跟左边的值比
        Gradient = rad - (axis_1 - Gra_range)
        # Gradient = Gradient / Gra_range * 0.5
        Gradient = Gradient / Gra_range
        print((rad - 7 * math.pi / 12) % (math.pi / 12))
        print("左Gradient值:", Gradient)
        # 左边
        # result = [(1.0 - Gradient) * colors[0][i] + Gradient * colors[1][i] for i in range(len(colors[0]))]
        result = [colors[0][i] + Gradient * colors[1][i] for i in range(len(colors[0]))]
        result = list(map(int, map(round, result)))
        return result
    # 标准绿
    if (axis_1 + Gra_range) <= rad < (axis_2 - Gra_range):
        result = colors[1]
        return result
    # 渐变 绿蓝 (225,255)
    if (axis_2 - Gra_range) <= rad < (axis_2 + Gra_range):
        # 右边 绿少蓝不变 跟右边的值比
        if rad - axis_2 > 0:
            Gradient = (axis_2 + Gra_range) - rad
            Gradient = Gradient / Gra_range * 0.5
            # Gradient = Gradient / Gra_range
            print((17 * math.pi / 12 - rad) % (math.pi / 12))
            print("右Gradient值:", Gradient)
            # result = [Gradient * colors[1][i] + (1.0 - Gradient) * colors[2][i] for i in range(len(colors[0]))]
            result = [Gradient * colors[1][i] + colors[2][i] for i in range(len(colors[0]))]
            result = list(map(int, map(round, result)))
            return result
        # 左边 绿不变蓝少 跟左边的值比
        Gradient = rad - (axis_2 - Gra_range)
        # Gradient = Gradient / Gra_range * 0.5
        Gradient = Gradient / Gra_range
        print((rad - 15 * math.pi / 12) % (math.pi / 12))
        print("左Gradient值:", Gradient)
        # result = [(1.0 - Gradient) * colors[1][i] + Gradient * colors[2][i] for i in range(len(colors[0]))]
        result = [colors[1][i] + Gradient * colors[2][i] for i in range(len(colors[0]))]
        result = list(map(int, map(round, result)))
        return result
    # 标准蓝
    if (axis_2 + Gra_range) <= rad < (axis_3 - Gra_range):
        result = colors[2]
        return result
    # 渐变 蓝不变红少 (345,360)
    if (axis_3 - Gra_range) <= rad < axis_3:
        Gradient = rad - (axis_3 - Gra_range)
        # Gradient = Gradient / Gra_range * 0.5
        Gradient = Gradient / Gra_range
        print((rad - 23 * math.pi / 12) % (math.pi / 12))
        print("左Gradient值:", Gradient)
        # result = [Gradient * colors[0][i] + (1.0 - Gradient) * colors[2][i] for i in range(len(colors[0]))]
        result = [Gradient * colors[0][i] + colors[2][i] for i in range(len(colors[0]))]
        result = list(map(int, map(round, result)))
        return result

# 下一步可能浮点数不能color(四舍五入解决) 用这个函数画条纹 可能要逐行像素 这个角度和角速度和相机刷新频率决定条纹宽度
# 要与相机频率和投影仪频率相结合
def create_Stripe(srcImg,color):
    height, width = srcImg.shape[:2]

    final_Img = cv.addWeighted(srcImg, 0.95, stripes, 0.05, 0)
    return final_Img, srcImg

if __name__ == "__main__":
    # 总时间
    t = 5
    # 角速度 rad/s 每秒的弧度   2 * math.pi     (一圈的弧度是2*pi)
    omega = 2
    # 初始相位 角度
    phase = 230
    # 角度
    angle = (omega * t + phase) % 360 # 90 180 270 360 取余
    print("angle值:",angle)

    # 每个角度一个颜色
    color = get_color(angle)
    print(color)  # 输出对应的颜色

    # circle = f = 旋转速度 = 每秒的圈数 = 一秒的总弧度 / 一圈的弧度(2*pi)
    circle = omega / 2 * math.pi
    # 三色轮的刷新率 Hz ,颜色序列的频率
    refresh_wheel = circle * 3

    # 相机cmos的读取速率 r ,每秒读取r次 设置为色轮的两倍
    r = circle * 2
    # 每秒单个条纹的总宽度  = 每种颜色持续时间 * 读取频率
    V_red_s = (1 / refresh_wheel) * r
    # 彩虹周期: 一个颜色旋转一周,
    # T = 1 / circle
    T = 2 * math.pi / omega # 弧度
    # 单个条纹的宽度
    V_red = T / (3 * r)
    # V_red = round(V_red)
    print(V_red)
    srcImg = cv.imread("G:/Image_Decomposition/RainbowNet_Data/data_pre/data_prepare/picture/Places365_test_00000001.jpg")








