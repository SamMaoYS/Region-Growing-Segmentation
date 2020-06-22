import cv2
from cv2 import ximgproc
import numpy as np
import common


class CoSLIC:
    def __init__(self, img, s=30, compactness=10, max_iter=10):
        self._img = img
        self._compactness = compactness
        self._max_iter = max_iter
        self._s = s
        self._labels = None
        self._n_labels = 0

    def slic(self):
        blur = common.gaussian_blur(self._img, sigma=0, size=3)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
        slic = ximgproc.createSuperpixelSLIC(lab, algorithm=ximgproc.SLIC, region_size=self._s, ruler=self._compactness)
        slic.iterate(num_iterations=self._max_iter)
        slic.enforceLabelConnectivity(min_element_size=self._s)
        mask = slic.getLabelContourMask(thick_line=False)
        self._labels = slic.getLabels()
        self._n_labels = slic.getNumberOfSuperpixels()
        mask_c3 = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        slic_viz = (mask_c3 == 0) * self._img
        mask_c3[:, :, 0:2] = 0
        slic_viz = slic_viz + mask_c3
        return slic_viz

    def canny_edge(self, low=0.1, high=0.2):
        blur = common.gaussian_blur(self._img, 0, 5)
        dx = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
        dy = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
        gradient = np.zeros_like(dx)
        cv2.magnitude(dx, dy, gradient)
        max_grad = np.amax(gradient)
        print(max_grad)
        return cv2.Canny(np.int16(dx), np.int16(dy), max_grad*low, max_grad*high)

