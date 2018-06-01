import cv2 as cv
import numpy as np
import sys

img = cv.imread("data/YaleB/yaleB02/yaleB02_P00A+000E+00.png",0)

if img is None:
    print("Error::Cannot read image")
    sys.exit()

print(img)

cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()