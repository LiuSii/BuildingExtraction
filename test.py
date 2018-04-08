import cv2 as cv

from function import IO
from function import Deal

src = IO.img_in("../NotPush/map.jpg")
IO.img_out("src", src)

# 转为hsv
hsv = cv.cvtColor(src, cv.COLOR_RGB2HSV)
h, s, v = Deal.split(hsv)

# IO.img_out("h", h)
# IO.img_out("s", s)
# IO.img_out("v", v)

# 突出显示H值
h, w = hsv.shape[:2]
for i in range(h):
    for j in range(w):
        hsv[i, j, 0] = 255
IO.img_out("test", hsv)

# k_means实现
k_img = Deal.k_means(hsv, 2)
IO.img_out("k_means", k_img)

# 二值化
k_img_gray = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
IO.img_out("gray", k_img_gray)
ret0, shadow_2value = cv.threshold(k_img_gray, 60, 255, cv.THRESH_BINARY)
IO.img_out("shadow_2value", shadow_2value)
print(shadow_2value.shape)
print(src.shape)

# 消除阴影与绿植
#img = cv.bitwise_and(shadow_2value, src)
#IO.img_out("NoGreenAndShadow", img)



cv.waitKey(0)
cv.destroyAllWindows()