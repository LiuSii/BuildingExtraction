"""
    毕业设计
    基于主动轮廓的卫星图像建筑物提取

"""
import cv2 as cv
import numpy as np


def img_in(str_path):
    # 图片输入
    image = cv.imread(str_path)
    return image


def img_out(win_name, image):
    # 图片输出
    cv.imshow(win_name, image)


def split(image):
    # 分量的提取
    ch1, ch2, ch3 = cv.split(image)
    return ch3, ch2, ch1


# 输入输出原图
img = img_in("resources/map2.jpg")
img_out("original", img)

# RGB输出
B, G, R = split(img)
"""
img_out("R", R)
img_out("G", G)
img_out("B", B)
"""

# RGB转HSV
hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
img_out("hsv", hsv)

# HSV输出
H, S, V = split(hsv)
"""
img_out("H", H)
img_out("S", S)
img_out("V", V)
"""

# 读入
Z = img.reshape((-1, 3))

# 转换为float类型
Z = np.float32(Z)

# 定义标准、K值
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4

# 调用k_means函数
ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
# 转回uint8格式，恢复图像
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv.imshow("k-cluster", res2)


# 等待
cv.waitKey(0)
cv.destroyAllWindows()