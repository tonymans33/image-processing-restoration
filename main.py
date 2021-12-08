import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal.signaltools import wiener
import skimage
from scipy import ndimage


def add_gaussian_noise(img, sigma):
    s = sigma
    gauss = np.random.normal(0, sigma, img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
    img_gauss = cv.add(img, gauss)
    return img_gauss


def add_salt_paper_noise(img):
    img_sp = skimage.util.random_noise(img, mode='s&p')
    return img_sp


# reading img
img = cv.imread("man.png")
cv.imshow("before gaussian noise", img)


# applying gaussian noise
im_gaussian_noise = add_gaussian_noise(img, 1)
cv.imshow("after gaussian noise", im_gaussian_noise)


# applying wiener filter
filtered_img_wiener = wiener(im_gaussian_noise)
cv.imshow("after wiener filter", filtered_img_wiener)


# applying salt paper noise
im_sp_noise = add_salt_paper_noise(img)
cv.imshow("after sp noise", im_sp_noise)


# applying mean filter




# applying median filter
img_median_filter = ndimage.median_filter(img, size=5)
cv.imshow("after median filter", img_median_filter)


# applying max filter
# Reduces pepper noise " find brightest"
img_max_filter = ndimage.maximum_filter(img, size=5)
cv.imshow("after max filter", img_max_filter)


# applying min filter
# Reduces pepper noise " find darkest"
img_min_filter = ndimage.minimum_filter(img, size=5)
cv.imshow("after min filter", img_min_filter)

cv.waitKey(0)
