import cv2 as cv

src = cv.imread("./resources/yuan.jpg")
cv.imshow("11", src)
print(src)

img = src[29:200, 42:214]
cv.imshow("22", img)

cv.waitKey(0)
cv.destroyAllWindows()
