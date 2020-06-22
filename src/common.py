import cv2
import glob
import numpy as np


def gaussian_blur(img, sigma=np.sqrt(2), size=-1):
    k_size = size
    if k_size == -1:
        k_size = np.int(2 * np.ceil(4 * sigma) + 1)
    blur = cv2.GaussianBlur(img, (k_size, k_size), sigma)
    return blur


def image_show(img, name="Image", wait=False):
    cv2.imshow(name, img)
    if wait:
        cv2.waitKey(0)


def image_load(path):
    images = [cv2.imread(file) for file in glob.glob(path)]
    return images
