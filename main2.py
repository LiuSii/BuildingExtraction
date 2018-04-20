"""
    毕业设计
    基于主动轮廓的卫星图像建筑物提取
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 导入函数文件
from function import Deal
from function import Algo
from function.Algo import Point


"""===================================================图像预操作==================================================="""
'''----------------------输入输出原图----------------------'''
src = cv.imread("../NotPush/map2.jpg")
# # cv.imshow("src", src)
# '''---------------------提取各个色彩空间-------------------'''
B, G, R = Deal.split(src)  # RGB顺序相反
# 转为hsv
hsv = cv.cvtColor(src, cv.COLOR_RGB2HSV)
H, S, V = Deal.split(hsv)
# cv.imshow("V", V)

# 转换到要处理的图像
img = V
print(img)
print(img.shape)

"""===================================================初始曲线提取================================================="""

'''--------使用给定的单值图像表示并计算强度直方图----------'''
# hist = cv.calcHist([img], [0], None, [256], [0, 256])
#
# plt.figure()  # 新建一个图像
# plt.title("Histogram of img")  # 标题
# plt.xlabel("值")  # x轴名称与坐标范围
# plt.xlim([0, 256])
# plt.ylabel("数量")
#
# plt.plot(hist)  # 绘图
# plt.show()

'''------------------------设定参数------------------------'''
# 迭代参数
IterationNumber = 10  # em.....论文写的经验值
# 聚类数
ClusterNumber = 2  # 分为前景和背景两个聚类

'''--------------使用k_means++算法求k个质心-------------'''
# 初始化质心，根据聚类数确定个数
cluster_centers = [Point() for _ in range(ClusterNumber)]
print(cluster_centers)
# 将0值的聚类中心代入函数寻找k_means++的初始质心
cluster_centers = Algo.k_means_plus(img, cluster_centers)

'''-------------------重复聚类步骤直到稳定-----------------'''
# 基于距离聚类各点
# 在每一个聚类中计算新的质心点


'''--------使用矩形结构元素对前景图像进行形态学操作--------'''
# 计算连通部分或者对象的数量，并保存其长宽为Xi,Yi
# 使用检测物体的长宽平均值来计算结构元素SE的长宽值
# 使用SE进行腐蚀和膨胀操作

'''-----------------应用边界检测提取初始曲线---------------'''


'''----------------------输入输出原图----------------------'''

"""==================================================主动轮廓模型=================================================="""

"""====================================================最终结果===================================================="""

"""======================================================等待======================================================"""
cv.waitKey(0)
cv.destroyAllWindows()
