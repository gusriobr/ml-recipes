import itertools

import cv2
import numpy as np

"""
    HOUGH LINES TRANSFORMATION
"""


def draw_hough(img, lines):
    """
    Draws obtained lines from hough line methods (prob and not prob)
    :param img:
    :param lines:
    :param filter_by_angle: angle to set valid slope
    :return:
    """
    if lines is None:
        return
    for line in lines:
        if line.shape == (1, 2):  # hough
            for r, t in line:
                a = np.cos(t)
                b = np.sin(t)
                x0 = a * r
                y0 = b * r
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


def filter_by_slope(hough_lines, angle_range=[]):
    filtered_lines = []
    if hough_lines is None:
        return None

    for line in hough_lines:
        if line.shape == (1, 2):  # hough
            for r, t in line:
                angle = np.rad2deg(t)
                if angle >= angle_range[0] and angle <= angle_range[1]:
                    filtered_lines.append(line)
        else:
            for x1, y1, x2, y2 in line:
                # slope = (y2 - y1) / (x2 - x1)
                angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
                if angle >= angle_range[0] and angle <= angle_range[1]:
                    filtered_lines.append(line)

    return filtered_lines


def __convert_to_array(value):
    if not type(value) == type([]):  # list
        return [value]
    else:
        return value


def hough_parameters_grid():
    grado = np.pi / 180
    return hough_parameters(rho=[1, 3, 9, 27, 90],
                            theta=[grado, grado * 3, grado * 9, grado * 27, grado * 70],
                            threshold=[350])


class HoughParameters(object):
    pass


def hough_parameters(rho=1, theta=np.pi / 180, threshold=450, minLineLength=250, maxLineGap=20):
    """
    Parameter values for grid search over hough lines space
    :return:
    """
    ret_obj = HoughParameters()
    ret_obj.rho_values = __convert_to_array(rho)
    ret_obj.theta_values = __convert_to_array(theta)
    ret_obj.threshold_values = __convert_to_array(threshold)
    ret_obj.minLineLength = __convert_to_array(minLineLength)
    ret_obj.maxLineGap = __convert_to_array(maxLineGap)

    return ret_obj


def get_hough_lines_batch(img, edges, method="hough", tmp_folder="/tmp", parameters=None, angle_filter=None):
    p = parameters
    if p is None:
        p = hough_parameters_grid()

    if method == "hough":
        param_map = itertools.product(p.rho_values, p.theta_values, p.threshold_values)
    else:
        param_map = itertools.product(p.rho_values, p.theta_values, p.threshold_values, p.minLineLength, p.maxLineGap)

    img_list = []
    print ("folder;theta;rho;ths")
    for m in param_map:
        rho = m[0]
        theta = m[1]
        ths = m[2]
        if method == "hough":
            lines = cv2.HoughLines(edges, rho, theta, threshold=ths)
        elif method == "houghP":
            minLineLength = m[3]
            maxLineGap = m[4]
            lines = cv2.HoughLinesP(edges, rho, theta, ths, minLineLength, maxLineGap)
        else:
            raise Exception("Invalid 'method' parameter value, possible values are: 'hough', 'houghP")

        numlin = 0
        img_copy = img.copy()
        if angle_filter is not None:
            lines = filter_by_slope(lines, angle_filter)
        draw_hough(img_copy, lines)

        print ("{};{};{};{} ".format(theta, rho, ths, numlin))

        img_list.append(img_copy)
        cv2.imwrite("{}/{}_{}_{}_{}.png".format(tmp_folder, method, theta, rho, ths), img_copy)

        # stack up images to visually compare them
        img_full = np.vstack(img_list)
        cv2.imwrite("{}/{}_full.png".format(tmp_folder, method), img_full)
