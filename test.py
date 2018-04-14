from matplotlib import pyplot as plt

# def image_hist(image):
#     """
#         绘制三通道直方图
#         其中RGB的直方图分别为红绿蓝色；HSV的直返图分别为蓝绿红色
#         :param image: 三通道图像
#         :return: 直接绘制
#     """
#
#     # 绘制的颜色
#     colors = ('blue', 'green', 'red')
#     for i, color in enumerate(colors):
#         hist = cv.calcHist([image], [i], None, [256], [0,256])
#         plt.plot(hist, color=colors)
#         plt.xlim([0, 256])
#     plt.show()
