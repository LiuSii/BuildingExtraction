import cv2 as cv

def split(image):
    # 分量的提取
    ch1, ch2, ch3 = cv.split(image)
    return ch3, ch2, ch1