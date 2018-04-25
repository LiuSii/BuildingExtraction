import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from function.Algo import Point


def split(image):
    # 分量的提取
    ch1, ch2, ch3 = cv.split(image)
    return ch1, ch2, ch3


def k_means(image, k):
    # k_means实现，输入k值

    # k_means读入
    img_shape = image.reshape((-1, 3))
    # 转换为float类型
    data = np.float32(img_shape)
    # 定义标准
    criteria = (cv.TERM_CRITERIA_EPS, 10, 10.0)
    # 调用k_means函数
    ret, label, center = cv.kmeans(data, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # 转回uint8格式，恢复图像
    center = np.uint8(center)
    k_img_ori = center[label.flatten()]
    k_img = k_img_ori.reshape((image.shape))

    return k_img


def translate(image, src_image):
    # 去除阴影，所有像素遍历，黑白反转，与源图像srcIamge作加处理，实现将阴影变为白色
    # 第一个输入为反转图像，第二个为源图像

    # 遍历将图转为二值
    h, w, c = image.shape
    print(image.shape)
    for row in range(h):
        for col in range(w):
             if image[row, col, 0] == 0:
                image[row, col, 0] = 255
                image[row, col, 1] = 255
                image[row, col, 2] = 255
             else:
                image[row, col, 0] = 0
                image[row, col, 1] = 0
                image[row, col, 2] = 0

    # 与源图像进行加处理
    f_image = cv.add(image, src_image)
    print(image.shape)
    print(src_image.shape)

    return f_image


def canny(img):
    # canny边缘检测

    # 高斯模糊
    blurred = cv.GaussianBlur(img, (3, 3), 0)
    # 灰度图
    # gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)
    # 计算x、y方向梯度
    x_grad = cv.Sobel(blurred, cv.CV_16SC1, 1, 0)
    y_grad = cv.Sobel(blurred, cv.CV_16SC1, 0, 1)
    # canny边缘检测
    edge = cv.Canny(x_grad, y_grad, 50, 100)

    # 在原图基础上显示
    dst = cv.bitwise_and(img, img, mask=edge)

    return edge, dst


# 颜色区域提取
def color_area_red(self):
    # 提取红色区域(暂定框的颜色为红色)

    low_red = np.array([50, 60, 100])  # 为bgr
    high_red = np.array([85, 130, 180])
    mask = cv.inRange(self, low_red, high_red)
    red = cv.bitwise_and(self, self, mask=mask)
    return red


def color_area_blue(self):
    # 提取蓝色区域(暂定框的颜色为蓝色)

    low_red = np.array([90, 90, 60])  # 为bgr
    high_red = np.array([255, 220, 200])
    mask = cv.inRange(self, low_red, high_red)
    red = cv.bitwise_and(self, self, mask=mask)
    return red


def max_b(image):
    """
        取b值最大区域
    """

    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            if (image[row, col, 0] < image[row, col, 1] + 5) | (image[row, col, 0] < image[row, col, 2] + 5):  # 判断是否B最大且大一部分
                image[row, col, 0] = image[row, col, 1] = image[row, col, 2] = 0
    return image


def show_value(image, n):
    """
        突出显示某值
        R、G、B的n值分别为2、1、0；H、S、V的n值分别为0、1、2
    """

    h, w = image.shape[:2]
    for i in range(h):
        for j in range(w):
            image[i, j, n] = 255
    return image


def image_hist(image, masks):
    """
        绘制三通道直方图
        其中RGB的直方图分别为红绿蓝色；HSV的直返图分别为蓝绿红色
        :param image: 三通道图像
        :return: 直接绘制
    """

    # 绘制的颜色
    colors = ('blue', 'green', 'red')
    for i, color in enumerate(colors):
        hist = cv.calcHist([image], [i], masks, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


def point_out(point):
    """
    输出point类点的各值
    :param point: point类
    :return: 无
    """
    print("x:%s, y:%s, value:%s, group:%s" % (point.x, point.y, point.value, point.group))


def to_binary(img, flag):
    """
    将聚类结果变为二值化（当k值为2时）
    :param img: kmeans后的结果
    :param flag: 需要反转标志位，为0时不需要，为1时需要
    :return: 黑白二值化
    """
    width, height = img.shape  # 获取长宽

    # 判断标志位
    if flag == 0:
        swallow = 255
        deep = 0
    else:
        swallow = 0
        deep = 255

    for w in range(width):  # 遍历每一个点
        for h in range(height):
            if img[w][h] == 2:
                img[w][h] = deep
            else:
                img[w][h] = swallow
    return img
