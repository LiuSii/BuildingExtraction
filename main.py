"""
    毕业设计
    基于主动轮廓的卫星图像建筑物提取

"""
import cv2 as cv
import numpy as np

# 导入其他文件函数库
from function import IO
from function import Deal

# 输入输出原图
img = IO.img_in("resources/map2.jpg")
IO.img_out("original", img)

# RGB分离
B, G, R = Deal.split(img)
# RGB转HSV并输出
hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
# IO.img_out("hsv", hsv)
# HSV分离并输出
H, S, V = Deal.split(hsv)
"""
IO.img_out("H", H)
IO.img_out("S", S)
IO.img_out("V", V)
"""
# k_means读入
img_shape = img.reshape((-1, 3))
# 转换为float类型
data = np.float32(img_shape)
# 定义标准、K值
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 6
# 调用k_means函数
ret, label, center = cv.kmeans(data, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# 转回uint8格式，恢复图像
center = np.uint8(center)
k_img_ori = center[label.flatten()]
k_img = k_img_ori.reshape((img.shape))

# 消除阴影为白色
dst = Deal.translate(k_img)

cv.imshow("k-cluster", dst)


# 等待
cv.waitKey(0)
cv.destroyAllWindows()