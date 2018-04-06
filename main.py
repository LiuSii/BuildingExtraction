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
# img = IO.img_in("./resources/map2.jpg")  # 测试图
img = IO.img_in("../NotPush/map.jpg")
IO.img_out("original", img)

# k_means实现
k_img = Deal.k_means(img, 3)
IO.img_out("k_means", k_img)

# 二值化
ret0, shadow_2value = cv.threshold(k_img, 60, 255, cv.THRESH_BINARY)
IO.img_out("shadow_2value", shadow_2value)

# 消除阴影为白色，其他部分还原
NoShadow_img = Deal.translate(shadow_2value, img)
IO.img_out("NoShadow_img", NoShadow_img)

# 流出RGB中偏蓝色的区域
test = Deal.color_area_blue(NoShadow_img)
IO.img_out("test", test)

# 保留B值最大的区域
test_b = Deal.max_b(test)
IO.img_out("test_b", test_b)

# 输出一个图像
# cv.imwrite("../NotPush/out1.jpg", NoShadow_img)

# # 输出rgb
# r, g, b = Deal.split(img)
# IO.img_out("r", r)
# IO.img_out("g", g)
# IO.img_out("b", b)

# # 转为hsv
# hsv = cv.cvtColor(NoShadow_img, cv.COLOR_RGB2HSV)
#
# IO.img_out("hsv", hsv)
# V, S, H = Deal.split(hsv)
# # IO.img_out("H", H)
# IO.img_out("S", S)
# # IO.img_out("V", V)
#
# # 将S图二值化
# ret1, _2value = cv.threshold(S, 110, 255, cv.THRESH_BINARY)
# IO.img_out("2value", _2value)
# #
# # 对二值化的S图进行开运算
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
# opened = cv.morphologyEx(_2value, cv.MORPH_OPEN, kernel)
# IO.img_out("Open", opened)
#
# # 开运算后闭运算，优化线条
# closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel)
# # 显示腐蚀后的图像
# IO.img_out("Close", closed)
#
# # 轮廓边缘检测
# canny_out, img3 = Deal.canny(closed)
# IO.img_out("canny", canny_out)
#
# # 轮廓线加到原图上
# canny_out_color = cv.cvtColor(canny_out, cv.COLOR_GRAY2RGB)
# extraction1 = cv.add(canny_out_color, img)
# cv.imshow("extraction1", extraction1)

# 等待
cv.waitKey(0)
cv.destroyAllWindows()