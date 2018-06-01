__author__ = 'Cristobal Barrientos Low'
__email__  = 'cbarrientoslow@gmail.com'

import numpy as np
import cv2 as cv
import time

# Obtiene histograma de una imagen
def get_histogram(im):

    h = dict()
    for i in range(256):
        h[i] = 0

    for y in range(0,im.shape[0]):
        for x in range(0,im.shape[1]):
            key = int(im[y,x])
            if key in h:
                h[key] += 1

    return h

# Obtiene el valor de un bin de un histograma
def get_histogram_value(im, value):
    output = 0
    for y in range(0,im.shape[0]):
        for x in range(0,im.shape[1]):
            if im[y,x] == value:
                output += 1

    return output

# Obtiene el valor acumulado de un histograma hasta cierto nivel de bin
def get_accumulative_histogram_value(roi, value, levels):
    hist = [0] * levels
    for j in range(0, roi.shape[0]):
        for i in range(0, roi.shape[1]):
            hist[roi[j, i]] += 1

    return sum(hist[0:value + 1])

# Implementa la ecualización local
def local_equalization(w_height, w_width, im):
    #start_time = time.time()
    #print("Aplicando ecualización local con ventanas de tamaño (" + str(w_height) + ", " + str(w_width) + ")")
    #print(im.shape)

    levels = 256
    mn = float(w_height * w_width)
    height = im.shape[0]
    width = im.shape[1]
    factor = (levels - 1) / mn
    #print(factor)

    mid_w = int(w_width / 2.0)
    mid_h = int(w_height / 2.0)

    eq_im = np.zeros(im.shape, dtype=float)
    for y in range(0, height):
        for x in range(0, width):
            roi = im[y - mid_h: y + mid_h, x - mid_w: x + mid_w]
            eq_im[y, x] = factor * float(get_accumulative_histogram_value(roi, im[y, x], levels))

    #print("--- Ecualización local realizada en %s seconds ---" % (time.time() - start_time))

    return np.array(eq_im).astype(float)

# Implementa la normalización local
def local_normalization(w_height, w_width, im):
    #print("Aplicando normalización local con ventanas de tamaño (" + str(w_height) + ", " + str(w_width) + ")")
    #start_time = time.time()
    # Get integral images
    int_im, int_im_sq = cv.integral2(im)
    #print("Imagen integral obtenida")
    #print(int_im.shape)
    #print(int_im_sq.shape)

    #print(im.shape)
    height = im.shape[0]
    width = im.shape[1]
    w_area = float(w_height * w_width)
    norm_im = np.zeros(im.shape, dtype = float)

    mid_w = int(w_width / 2.0)
    mid_h = int(w_height / 2.0)
    # Recorriendo imagen original
    for y in range(0, height):
        for x in range(0, width):
            sumW = float(get_sum_for_window(x, y, w_height, w_width, int_im))
            sumWSQ = float(get_sum_for_window(x, y, w_height, w_width, int_im_sq))
            mean = sumW / w_area
            dvst = np.sqrt((sumWSQ/w_area) - (mean * mean))
            if dvst > 0:
                norm_im[y, x] = (float(im[y, x]) - mean) / dvst

    #print("--- Normalización local realizada en %s seconds ---" % (time.time() - start_time))
    #norm_im = (norm_im - norm_im.min())
    #norm_im = norm_im / (norm_im.max() - norm_im.min())

    return np.array(norm_im).astype(np.float)

# Obtiene la suma de una ventana de una imagen dada su imagen integral
def get_sum_for_window(x, y, w, h, summed_area):

    mid_w = int(w / 2)
    mid_h = int(h / 2)
    height = summed_area.shape[0]
    width = summed_area.shape[1]

    # Manage coordinates
    ax = x - mid_w
    if ax <= 0:
        ax = 0

    ay = y - mid_h
    if ay < 0:
        ay = 0

    bx = x + mid_w
    if bx >= width:
        bx = width - 1

    by = ay

    cx = bx

    cy = y + mid_h
    if cy >= height:
        cy = height - 1

    dx = ax

    dy = cy

    # Get summed values
    A = summed_area[ay, ax]
    B = summed_area[by, bx]
    C = summed_area[cy, cx]
    D = summed_area[dy, dx]

    return (C + A - B - D)
