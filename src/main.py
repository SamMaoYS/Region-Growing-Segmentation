import glob
import cv2
from cv2 import ximgproc
import numpy as np
from skimage import segmentation
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries


def img_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)


def slic_cv(img, compactness=10, m_iter=10, s=30):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    slic = ximgproc.createSuperpixelSLIC(lab, algorithm=ximgproc.SLIC, region_size=s, ruler=compactness)
    slic.iterate(m_iter)
    slic.enforceLabelConnectivity(s)
    mask = slic.getLabelContourMask()

    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    result = (mask == 0) * img
    mask[:, :, 0:2] = 0
    result = result + mask
    return result


def slic_ski(img, compactness=10, m_iter=10, s=30):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = segmentation.slic(img, n_segments=np.size(img) / (3*np.power(s, 2)), sigma=0.5, convert2lab=True, compactness=compactness,
                               enforce_connectivity=True, max_iter=m_iter)
    print(result)
    return result


def main():
    images = [cv2.imread(file) for file in glob.glob('../data/*.jpg')]
    img = images[2]

    compactness = 15
    m_iter = 10
    s = 30
    result1 = slic_cv(img, compactness, m_iter, s)
    result2 = slic_ski(img, compactness, m_iter, s)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(mark_boundaries(img, result2))
    plt.show()
    img_show("slic", result1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
