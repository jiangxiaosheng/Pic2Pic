import numpy as np


def otsu(gray_img):
    width, height = gray_img.shape
    pixel_number = width * height
    mean_weight = 1.0 / pixel_number
    best_thresh = -1
    final_value = -1
    hist, bins = np.histogram(gray_img, np.arange(0, 257))
    intensity_arr = np.arange(256)
    for t in bins[1:-1]:
        pcb = np.sum(hist[:t])
        pcf = np.sum(hist[t:])
        wb = pcb * mean_weight
        wf = pcf * mean_weight
        mub = np.sum(intensity_arr[:t] * hist[:t]) / pcb
        muf = np.sum(intensity_arr[t:] * hist[t:]) / pcf
        value = wb * wf * (mub - muf) ** 2
        if value > final_value:
            best_thresh = t
            final_value = value
    otsu_img = gray_img.copy()
    otsu_img[gray_img > best_thresh] = 255
    otsu_img[gray_img < best_thresh] = 0
    return otsu_img
