import cv2 as cv
# from random import *
import random
import numpy as np


# 类：点
class Point:
    # x和y是二维数组的坐标，value是其单通道值，group为该点的聚类类型
    __slots__ = ["x", "y", "value", "group"]

    # 初始化归零
    def __init__(self, x=0, y=0, value=0, group=0):
        self.x, self.y, self.value, self.group = x, y, value, group


# 最大距离：无限大
DISTANCE_MAX = 1e100


def p2p_distance(point_a, point_b):
    """
    求两个Point点之间的距离
    :param point_a: Point类点a
    :param point_b: Point类点b
    :return: 距离平方
    """
    # bug记录1：两个value都是无符号数，无符号数直接相减肯定会出现问题，所以先判断大小
    if point_a.value > point_b.value:
        result = point_a.value - point_b.value
    else:
        result = point_b.value - point_a.value
    return result ** 2


def nearest_center(point, cluster_centers):
    """
    求一个点的最近中心点
    :param point: 目标点，Point类
    :param cluster_centers: 各个中心点，为一个list，输入之前的已有初始点，list内元素为Point类
    :return: 最小索引和最小距离
    """

    # 设定最大最小值
    min_index = point.group
    min_dist = DISTANCE_MAX

    # 递归比较，找到最小的那个中心点
    for i, p in enumerate(cluster_centers):  # i为在cluster_centers里的索引，p为该Point点
        dis = p2p_distance(p, point)
        if min_dist > dis:
            min_dist = dis
            min_index = i

    return min_index, min_dist


def k_means_plus(img, cluster_centers):
    """
    Kmeans++初始化中心点函数
    :param img:数据集，二维数组，或者是Mat形式，单通道
    :param cluster_centers: 聚类初始点，是一个list
    :return:返回List存放的k个初始点
    """
    # 获取图像宽高
    width, height = img.shape
    # 随机选取一个为中心点
    x0 = random.randint(0, width-1)
    y0 = random.randint(0, height-1)
    # 将随机值赋给初始第一个Point点的x和y值
    cluster_centers[0].x, cluster_centers[0].y = x0, y0
    cluster_centers[0].value = img[x0][y0]  # 为初始点一号赋value值
    print(cluster_centers[0].x, cluster_centers[0].y, cluster_centers[0].value)

    # 初始化存数每个数据点到最近中心点的距离的Mat数组
    distance = np.zeros([width, height], np.uint16)  # 定义一个同等大小的数组，初始归0

    # 递归求初始化
    for i in range(1, len(cluster_centers)):  # k-1次递归，k为聚类数
        cc_sum = 0  # 保存每个中心点的总和

        for w in range(width):  # 遍历每一个点
            for h in range(height):
                temporarily_point = Point(w, h, img[w][h])  # 取到目前点的坐标和值
                distance[w][h] = nearest_center(temporarily_point, cluster_centers[:i])[1]  # 计算该点离最近中心点的距离并储存
                cc_sum += distance[w][h]  # 存储总距离

        print(distance)
        print(cc_sum)

    #     # 概率化
    #     cc_sum *= random()
    #
    #     for j, dis in enumerate(distance):
    #         cc_sum -= dis
    #         if cc_sum > 0:
    #             continue
    #         cluster_centers[i] = copy(img[i])
    #         break
    # for p in img:
    #     p.group = nearest_center(p, cluster_centers)[0]
