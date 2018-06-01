#!/usr/bin/env python3
# Usage:: python3 op_image.py <image_path> <window_height> <window_width>
# Author: Cristobal Barrientos Low

__author__ = 'Cristobal Barrientos Low'
__email__  = 'cbarrientoslow@gmail.com'

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys
import time
import localnormutil as ln

print("opencv version: " + cv.__version__)

# Convert raw image (floats) into a image in range 0 - 255 (int)
# To save
def raw_to_image(raw_image):
    raw_image = (raw_image - raw_image.min())
    raw_image = raw_image / (raw_image.max() - raw_image.min())
    raw_image *= 255.
    return raw_image.astype(np.int)

def main():

    print("Usage:: python3 op_image.py <image_path> <window_height> <window_width>")

    # Set default values
    w_width = 3
    w_height = 3

    # Load image
    print(sys.argv)
    if len(sys.argv) <= 1:
        print("Error::You must set an image as a parameter")
        sys.exit()

    if len(sys.argv) > 2 and int(sys.argv[2]) > 0:
        w_height = int(sys.argv[2])

    if len(sys.argv) > 3 and int(sys.argv[3]) > 0:
        w_width = int(sys.argv[3])

    image_path = sys.argv[1]
    img = cv.imread(image_path,0)

    if img is None:
        print("Error::Cannot read image")
        sys.exit()

    print("Operando con ventanas de tamaño (" + str(w_height) + ", " + str(w_width) + ")")
    print(img.shape)
    print(img)
    #cv.imshow('Imagen de entrada',img / 255.0)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    hn_img = ln.local_normalization(w_height, w_width, img)

    print("Raw local normalized image")
    print(hn_img.shape)
    print(hn_img)

    eq_img = ln.local_equalization(w_height,w_width,img)

    print("Raw local equalized image")
    print(eq_img.shape)
    print(eq_img)


    hn_img = raw_to_image(hn_img)
    eq_img = raw_to_image(eq_img)

    print("Local normalized image")
    print(hn_img.shape)
    print(hn_img)

    print("Local equalized image")
    print(eq_img.shape)
    print(eq_img)


    # Write results to file
    vis = np.concatenate((img, hn_img, eq_img), axis=1)
    cv.imwrite('out.png', vis)

    # Get histograms
    h1 = ln.get_histogram(img)

    plt.plot(list(h1.keys()), list(h1.values()), 'b')
    plt.xlabel('Color Level')
    plt.ylabel('Count')

    h2 = ln.get_histogram(eq_img)
    plt.plot(list(h2.keys()), list(h2.values()), 'r')

    h3 = ln.get_histogram(hn_img)
    plt.plot(list(h3.keys()), list(h3.values()), 'g')

    plt.show()

main()