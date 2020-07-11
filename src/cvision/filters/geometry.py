import numpy as np
import cv2


def get_rectangle_slope(box):
    """
    :param box: four point tuple
    :return: float with slope of the longest sides referring to horizontal axis
    """
    # get longest line

    # calculate slope
