import cv2 as cv
import random
import numpy as np
from function import Deal

import sys


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
    k_means++初始化中心点函数
    :param img:数据集，二维数组，或者是Mat形式，单通道
    :param cluster_centers: 聚类初始点，是一个list，list内是Point类
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
    # print(cluster_centers[0].x, cluster_centers[0].y, cluster_centers[0].value)

    # 初始化存数每个数据点到最近中心点的距离的Mat数组
    distance = np.zeros([width, height], np.uint16)  # 定义一个同等大小的数组，初始归0

    # 递归求初始化
    for i in range(1, len(cluster_centers)):  # k-1次递归，k为聚类数
        # 计算每个点到中心质点的距离并储存
        # bug记录2：好像是有符号数达到了最大尺寸的一半会导致溢出，所以改用numpy的ulonglong类型
        cc_sum = np.ulonglong(0)  # 保存每个中心点的总和

        for w in range(width):  # 遍历每一个点
            for h in range(height):
                temporarily_point = Point(w, h, img[w][h])  # 取到目前点的坐标和值
                distance[w][h] = nearest_center(temporarily_point, cluster_centers[:i])[1]  # 计算该点离最近中心点的距离并储存
                cc_sum += distance[w][h]  # 存储总距离

        # print(distance)
        # print(cc_sum)

        # 化用公式(distance[i]/cc_sum)，并用一个0到1之间的随机数遍历进行寻找
        random_cc_sum = cc_sum * random.random()  # 化用第一步，直接给总数乘以0到1的随机数
        # print("random_cc_sum:%s" % random_cc_sum)

        break_flag = False  # 跳出多层循环的标志
        for w2 in range(width):  # 再次遍历每一个点
            for h2 in range(height):
                random_cc_sum -= distance[w2][h2]  # 如果小于0说明在这个区间内
                if random_cc_sum > 0:
                    continue
                else:
                    cluster_centers[i] = Point(w2, h2, img[w2][h2])  # 获得其点存入质心数组
                    break_flag = True
                    break
            if break_flag:
                break

    return cluster_centers


def k_means(src, k, cluster_centers, iteration_number=0, type_flag=0):
    """
    k_means单通道聚类
    :param src: 输入图像
    :param k: 聚类数
    :param cluster_centers:
    :param iteration_number: 聚类次数，默认为0，表示不使用聚类次数
    :param type_flag: 输入图像的类型，V值为0， S值为1
    :return: 返回聚类后的图像
    """
    changed = True  # 以聚类点是否变换判断是否收敛
    # 获取图像宽高
    width, height = src.shape
    # 初始化存数每个数据点聚类类型的Mat数组
    groups = np.zeros([width, height], np.uint8)  # 定义一个同等大小的数组，初始归0

    #print(width, height)

    # 先将初始聚类点存入聚类类型数组，并且加1存储（与初始值区别）
    for i, p in enumerate(cluster_centers):
        groups[p.x][p.y] = i + 1

    # print(groups)
    # print(cluster_centers[0].group, cluster_centers[1].group)

    # 没有迭代次数时设置标志
    if iteration_number == 0:
        iteration_number_flag = 1
    else:
        iteration_number_flag = 0

    # 要么迭代次数到达，要么不再变化
    while changed & (bool(iteration_number_flag) | ((not iteration_number_flag) & bool(iteration_number))):
        changed = False
        iteration_number -= 1

        # 对于每一个点计算离其最近的聚类
        for w in range(width):
            for h in range(height):
                temporarily_point = Point(w, h, src[w][h], groups[w][h])  # 取到目前点的坐标、值和聚类类型

                min_index2 = groups[w][h]  # 取此点原始距离
                min_dist2 = DISTANCE_MAX  # 准备存储最小距离

                # 递归比较，找到最小的那个中心点索引
                for i, p in enumerate(cluster_centers):  # i为在cluster_centers里的索引，p为该Point点
                    dis2 = p2p_distance(p, temporarily_point)
                    if min_dist2 > dis2:
                        min_dist2 = dis2
                        min_index2 = i+1

                # 判断索引是否有变化
                if groups[w][h] != min_index2:
                    changed = True
                    groups[w][h] = min_index2

        # 更新每个聚类的中心点
        for k_i in range(k):
            v_sum = 0  # 存储值总和
            v_count = 0  # 聚类总数

            # 同一聚类值相加
            for w2 in range(width):
                for h2 in range(height):
                    if k_i + 1 == groups[w2][h2]:
                        v_sum += src[w2][h2]
                        v_count += 1

            # 更新中心点
            center_value = v_sum/v_count
            cluster_centers[k_i].value = center_value

    # 保证浅色为白色，深色为黑色
    if cluster_centers[0].value < cluster_centers[1].value:  # 用[0]存储深色类
        binary_inv_flag = 0  # 反转标志正常
    else:
        binary_inv_flag = 1  # 反转标志异常，需要反转

    # # 根据输入图像类型判断反转
    # if type_flag == 1:  # 若为S图，得再反转，保证[0]为深色类
    #     binary_inv_flag = type_flag - binary_inv_flag

    return groups, binary_inv_flag
