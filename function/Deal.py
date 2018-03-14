import cv2 as cv
import numpy as np


def split(image):
    # 分量的提取
    ch1, ch2, ch3 = cv.split(image)
    return ch3, ch2, ch1


def k_means(image, k):
    # k_means实现，输入k值

    # k_means读入
    img_shape = image.reshape((-1, 3))
    # 转换为float类型
    data = np.float32(img_shape)
    # 定义标准、K值
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 调用k_means函数
    ret, label, center = cv.kmeans(data, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # 转回uint8格式，恢复图像
    center = np.uint8(center)
    k_img_ori = center[label.flatten()]
    k_img = k_img_ori.reshape((image.shape))

    return k_img


def translate(image, src_image):
    # 去除阴影，所有像素遍历，近黑色转为黑色，其他转为白色，与源图像srcIamge作加处理，实现将阴影变为白色

    # 遍历将图转为二值
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
             if (image[row, col, 0] < 60) & (image[row, col, 1] < 60) & (image[row, col, 2] < 60):
                image[row, col, 0] = 255
                image[row, col, 1] = 255
                image[row, col, 2] = 255
             else:
                image[row, col, 0] = 0
                image[row, col, 1] = 0
                image[row, col, 2] = 0

    # 与源图像进行加处理
    f_image = cv.add(image, src_image)

    return f_image
