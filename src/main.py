import common
from coslic import CoSLIC
import cv2


def main():
    images = common.image_load('../data/*.jpg')
    for img in images:
        slic = CoSLIC(img)
        slic_viz = slic.slic()
        edges = slic.canny_edge()
        common.image_show(slic_viz, "slic")
        common.image_show(edges, "edges")

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
