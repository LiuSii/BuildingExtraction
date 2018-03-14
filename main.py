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

"""
# RGB分离
B, G, R = Deal.split(img)
# RGB转HSV并输出
hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
# IO.img_out("hsv", hsv)
# HSV分离并输出
H, S, V = Deal.split(hsv)

IO.img_out("H", H)
IO.img_out("S", S)
IO.img_out("V", V)
"""
# k_means实现
k_img = Deal.k_means(img, 6)

# 消除阴影为白色，其他部分还原
dst = Deal.translate(k_img, img)

cv.imshow("k-cluster", dst)

# 等待
cv.waitKey(0)
cv.destroyAllWindows()