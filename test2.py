import cv2


img = cv2.imread("./resources/test_map6.jpg")
h, w, _ = img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 寻找轮廓
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for c_i, contour in enumerate(contours):
    cv2.drawContours(img, contours, c_i, (0, 255, 0), 1)
cv2.imshow("contours", img)

# 需要搞一个list给cv2.drawContours()才行
c_max = []
for i in range(len(contours)):
    cnt = contours[i]
    area = cv2.contourArea(cnt)

    # 处理掉小的轮廓区域，这个区域的大小自己定义。
    if (area < (h / 10 * w / 10)):
        c_min = []
        c_min.append(cnt)
        # thickness不为-1时，表示画轮廓线，thickness的值表示线的宽度。
        cv2.drawContours(img, c_min, -1, (0, 0, 0), thickness=-1)
        continue

    c_max.append(cnt)

cv2.drawContours(img, c_max, -1, (255, 255, 255), thickness=-1)

cv2.imshow('image', img)
cv2.waitKey(0)
