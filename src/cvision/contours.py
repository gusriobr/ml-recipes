import cv2
import numpy as np
import ocvtools.image.imutils as imu

def contour_skim_to_ocv(contour):
    c_ocv = np.zeros(contour.shape, dtype=np.int32)
    # convertir coordenadas
    c_ocv[:, 0] = contour[:, 1]
    c_ocv[:, 1] = contour[:, 0]
    return c_ocv


def create_contour_mask(contour, size):
    mask = np.zeros(size, dtype = np.uint8)
    # draw contour on mat
    cv2.drawContours(mask, [contour], 0, (255), -1)

    return mask


def smooth_contour(img, ksize=7, iters=5, min_th = 1):

    th, mask = cv2.threshold(img,min_th, 255, cv2.THRESH_BINARY)
    blurred = cv2.pyrUp(mask)
    for i in range(0,iters):
        blurred = cv2.medianBlur(blurred,ksize)
    blurred = cv2.pyrDown(blurred)
    th, mask = cv2.threshold(blurred,180, 255, cv2.THRESH_BINARY)
    return mask

def get_mask_from_contour(contour, shape):
    """
    Creates a binary mask for the given contour using a mat with dimensions set with shape parameter
    :param contour:
    :param shape: tuple
    :return:
    """

    if len(shape) == 2:
        w, h = shape
    elif len(shape) == 3:
        w, h = shape[:2]
    else:
        raise ValueError("Invalid image shape, cannot create imagen numpy array.")

    mat = np.zeros((w,h), np.uint8)

    if contour is not None:
        mat = cv2.drawContours(mat, [contour], 0, (255), -1)

    return mat


def get_max_contour(frame, min_area = -1):
    """
    Finds the largest contour in the image
    :param frame: image to look into
    :param min_area: minimun area of the viable contours
    :return:
    """
    img = frame.copy()
    if imu.isColorImage(img):
        # get B/N image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)[1]

    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)

    # get just max contour
    if len(contours) == 0:
        return None
    elif len(contours) == 1:
        return contours[0]
    else:
        max_contour = None
        filtered_contours = []
        max_contour_position = -1
        max_contour_area = -1
        idx = 0
        for c in contours:
            cnt_area = cv2.contourArea(c)
            if cnt_area > min_area:
                if cnt_area > max_contour_area:
                    max_contour_area = cnt_area
                    max_contour_position = idx
                filtered_contours.append(c)
                idx += 1
        if len(filtered_contours) > 0:
            max_contour = filtered_contours[max_contour_position]
        return max_contour