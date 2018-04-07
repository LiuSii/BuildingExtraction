import cv2 as cv

from function import IO
from function import Deal

src = IO.img_in("./resources/money.png")
IO.img_out("src", src)

# 转为hsv
hsv = cv.cvtColor(src, cv.COLOR_RGB2HSV)
h, s, v = Deal.split(hsv)

IO.img_out("h", h)
IO.img_out("s", s)
IO.img_out("v", v)

h, w = src.shape[:2]
for i in range(h):
    for j in range(w):
        src[i, j, 2] = 255
IO.img_out("test", src)

cv.waitKey(0)
cv.destroyAllWindows()