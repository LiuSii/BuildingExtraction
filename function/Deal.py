import cv2 as cv


def split(image):
    # 分量的提取
    ch1, ch2, ch3 = cv.split(image)
    return ch3, ch2, ch1


def translate(image):
    # 去除阴影，所有像素遍历
    h, w, c= image.shape
    for row in range(h):
        for col in range(w):
             if (image[row, col, 0] < 60) & (image[row, col, 1] < 60) & (image[row, col, 2] < 60):
                image[row, col, 0] = 255
                image[row, col, 1] = 255
                image[row, col, 2] = 255
    return image
