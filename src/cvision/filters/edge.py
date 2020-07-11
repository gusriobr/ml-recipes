import cv2
import numpy as np

import ocvtools.image.imutils as imu
import ocvtools.video.vidutils as vut
from pigsty.segmentation.contours import get_contour_mask, get_max_contour


def apply_laplacian(img):
    # check if image is already BN
    if len(img.shape) > 2:
        bn_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        bn_image = img
    lap = cv2.Laplacian(bn_image, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    return lap


def apply_filter_by_contour(img, min_area=5600, draw_contour=True, fill_holes=False):
    cnt = get_max_contour(img, min_area=min_area)
    contour_mask = get_contour_mask(cnt, img.shape[:2])

    # contour_mask = smooth_contours(img, contour_mask)

    if cnt is not None:
        if fill_holes:
            # transfor to RGB mode
            if len(img.shape) > 2:
                img = cv2.drawContours(img, [cnt], 0, (255, 255, 255), cv2.FILLED)
            else:
                img = cv2.drawContours(img, [cnt], 0, 255, cv2.FILLED)

        if draw_contour and not imu.isColorImage(img):
            # transfor to RGB mode
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)

    if contour_mask is not None:
        img = cv2.bitwise_and(img, img, mask=contour_mask)

    return img


def apply_find_contours(img):
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(im3, contours, -1, (0, 255, 0), 3)
    return im3


def apply_blur(image, type="Gaussian", kernel=(5, 5)):
    if type.lower() == "gaussian":
        image = cv2.GaussianBlur(image, kernel, 0)
    else:
        raise Exception("Not implemented yet!!")
    return image


def apply_canny(image, minVal=80, maxVal=150):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(image, minVal, maxVal)
    return canny


def apply_sobel(image, type="both"):
    if "x" == type.lower() or "both" == type.lower():
        sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        sobelX = np.uint8(np.absolute(sobelX))
        img = sobelX
    if "y" == type.lower() or "both" == type.lower():
        sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)
        sobelY = np.uint8(np.absolute(sobelY))
        img = sobelY

    if "y" == type.lower() or "both" == type.lower():
        img = cv2.bitwise_or(sobelX, sobelY)

    return img


def smooth_contours(img, contour):
    from scipy.signal import savgol_filter

    # Use Savitzky-Golay filter to smoothen contour.
    window_size = int(
        round(min(img.shape[0], img.shape[1]) * 0.05))  # Consider each window to be 5% of image dimensions
    x = savgol_filter(contour[:, 0, 0], window_size * 2 + 1, 3)
    y = savgol_filter(contour[:, 0, 1], window_size * 2 + 1, 3)

    approx = np.empty((x.size, 1, 2))
    approx[:, 0, 0] = x
    approx[:, 0, 1] = y
    approx = approx.astype(int)
    contour = approx


if __name__ == '__main__':
    video_path = '/home/gus/workspaces/wks-python/pigspy/resources/v_test.avi'

    # # apply laplacian to video
    # metrics = vut.process_frames(video_path, apply_laplacian, show_video=False, start_second=2, end_second=14,
    #                              output_video='/home/gus/workspaces/wks-python/pigspy/resources/v_laplacian.avi')
    # # apply sobel to video
    # metrics = vut.process_frames(video_path, apply_sobel, show_video=False, start_second=2, end_second=14,
    #                              output_video='/home/gus/workspaces/wks-python/pigspy/resources/v_sobel.avi')
    #
    # apply canny to video
    metrics = vut.process_frames(video_path, apply_sobel, show_video=False, start_second=2, end_second=14,
                                 output_video='/home/gus/workspaces/wks-python/pigspy/resources/v_canny.avi')


    # img_path = '/home/gus/workspaces/wks-python/pigspy/resources/frames/frame320.jpg'
    # image = cv2.imread(img_path)
    # bn_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # lap = cv2.Laplacian(bn_image, cv2.CV_64F)
    # lap = np.uint8(np.absolute(lap))
    #
    # # cv2.imshow("asdf", lap)
    # # cv2.waitKey(0)
    #
    # cv2.imwrite("/tmp/laplacian.png", lap)
