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
img = IO.img_in("resources/map.jpg")
IO.img_out("original", img)

# k_means实现
k_img = Deal.k_means(img, 2)
IO.img_out("k_means", k_img)

# 二值化
ret0, shadow_2value = cv.threshold(k_img, 97, 255, cv.THRESH_BINARY)
IO.img_out("shadow_2value", shadow_2value)

# 消除阴影为白色，其他部分还原
NoShadow_img = Deal.translate(shadow_2value, img)
IO.img_out("NoShadow_img", NoShadow_img)

# 转为hsv
hsv = cv.cvtColor(NoShadow_img, cv.COLOR_RGB2HSV)

# IO.img_out("hsv", hsv)
V, S, H = Deal.split(hsv)
# IO.img_out("H", H)
IO.img_out("S", S)
# IO.img_out("V", V)

# 将S图二值化
ret, out_frame = cv.threshold(S, 110, 255, cv.THRESH_BINARY)
IO.img_out("frame", out_frame)
#
# 对二值化的S图进行开运算
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
opened = cv.morphologyEx(out_frame, cv.MORPH_OPEN, kernel)
IO.img_out("7", opened)

# 开运算后闭运算，优化线条
closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel)
# 显示腐蚀后的图像
IO.img_out("Close", closed)

# 轮廓边缘检测
img2, img3 = Deal.canny(closed)
IO.img_out("red_house_brim", img2)

# 轮廓线加到原图上
img4 = cv.cvtColor(img2, cv.COLOR_GRAY2RGB)
img5 = cv.add(img4, img)
print(img.shape)
print(img4.shape)
cv.imshow("1", img)
cv.imshow("11", img4)
cv.imshow("lll", img5)

# 等待
cv.waitKey(0)
cv.destroyAllWindows()