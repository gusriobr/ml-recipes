import cv2
import numpy as np
from sklearn.cluster import estimate_bandwidth, MeanShift

import ocvtools.image.imutils as imu


def apply_color_space(img, tag):
    if "HUE" in tag:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[:, :, 0]  # get hue channel
    elif "SAT" in tag:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[:, :, 1]  # get hue channel
    elif "HSV" in tag:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif "CIELAB" in tag:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    elif "RGB" in tag:
        # default color space
        pass
    elif "BN" in tag:
        # default color space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def apply_filter_by_GChannel(img):
    im2 = img.copy()
    im2[(im2[:, :, 1] > im2[:, :, 0]) & (im2[:, :, 1] > im2[:, :, 2])] = 0
    return im2


def apply_bn(img, range):
    if len(img.shape) > 2:
        raise Exception("This method can be applied only in BN images.")
    img2 = img.copy()

    mask = np.zeros(img2.shape[:2], np.uint8)
    mask[(img >= range[0]) & (img <= range[1])] = 255

    ret = cv2.bitwise_and(img, img, mask=mask)
    return ret


def apply_filter_channel(img, channel=0, range=[], mode="select"):
    c = img[:, :, channel]
    mask = np.zeros(img.shape[:2], np.uint8)
    value = 255
    if mode == "exclude":
        mask[:] = 255
        value = 0
    mask[(c >= range[0]) & (c <= range[1])] = value
    ret = cv2.bitwise_and(img, img, mask=mask)
    return ret


def apply_filter_2d_channels(img, channel1=0, range1=[], channel2=1, range2=[], mode="select"):
    c1 = img[:, :, channel1]
    c2 = img[:, :, channel2]
    mask = np.zeros(img.shape[:2], np.uint8)
    value = 255
    if mode == "exclude":
        mask[:] = 255
        value = 0
    mask[(c1 >= range1[0]) & (c1 <= range1[1]) & (c2 >= range2[0]) & (c2 <= range2[1])] = value
    ret = cv2.bitwise_and(img, img, mask=mask)
    return ret


def apply_hsv_filtering_by_range(img, range1=[], range2=[], convert_to_hsv=True):
    img2 = img.copy()
    if convert_to_hsv:
        hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    mask = np.zeros(img2.shape[:2], np.uint8)
    hue_channel = hsv[:, :, 0]

    mask[(hue_channel >= range1[0]) & (hue_channel <= range1[1])] = 255
    if range2:
        mask[(hue_channel >= range2[0]) & (hue_channel <= range2[1])] = 255

    ret = cv2.bitwise_and(img, img, mask=mask)
    return ret


def apply_hsv_filtering(img, hue_value=None, hue_range=None, sat_range=None, convert_to_hsv=True, fill_holes=True):
    if not hue_value and hue_range is None:
        raise Exception("One of these parameters must be set hue or hue_range must.")

    if hue_value > 180:
        raise Exception("Incorrect value for Hue parameter, opencv uses Hue in [0-180 range]")

    img2 = img.copy()
    if convert_to_hsv:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    if not imu.isColorImage(img2):
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    mask = np.zeros(img2.shape[:2], np.uint8)
    hue_channel = img2[:, :, 0]

    if hue_range is not None:
        mask[(hue_channel >= hue_range[0])
             & (hue_channel <= hue_range[1])] = 255
    else:
        mask[hue_channel >= hue_value] = 255

    ret = cv2.bitwise_and(img, img, mask=mask)

    if sat_range is not None:
        mask_sat = np.zeros(img2.shape[:2], np.uint8)
        mask_sat[img2[:, :, 0] < sat_range[0]] = 0
        mask_sat[img2[:, :, 0] > sat_range[1]] = 0
        mask_sat[img2[:, :, 0] != 0] = 255

        ret = cv2.bitwise_and(ret, ret, mask=mask)

    return ret


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    if d > 1:
        image = np.zeros((w, h, d), dtype=np.uint8)
    else:  # gray image
        image = np.zeros((w, h), dtype=np.uint8)

    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


# def quantization_kne

def quantization_meanshift_ocv(img, rad_space=30, rad_color=30):
    if len(img.shape) < 3:
        raise Exception("Opencv meanshift can be applied only on color images.")
    im2 = cv2.pyrMeanShiftFiltering(img, sp=rad_space, sr=rad_color)
    return im2


def quantization_meanshift(img, bandwidth=None):
    if len(img.shape) == 2:
        w, h = original_shape = tuple(img.shape)
        image_array = np.reshape(img, (w * h))
        image_array = image_array.reshape([-1, 1])
    else:
        w, h, d = original_shape = tuple(img.shape)
        image_array = np.reshape(img, (w * h, d))

    # The following bandwidth can be automatically detected using
    if not bandwidth:
        bandwidth = estimate_bandwidth(image_array, quantile=0.2, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(image_array)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    im2 = recreate_image(cluster_centers, labels, w, h)

    return im2


def quantization_kmeans(img, n_colors=5, num_iters=10, epsilon=1):
    img_deepness = 1
    if len(img.shape) > 2:
        img_deepness = 3

    Z = img.reshape((-1, img_deepness))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, num_iters, epsilon)
    ret, label, center = cv2.kmeans(Z, n_colors, None, criteria, num_iters, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2


def quantization_ked(img):
    """
    Imagen should be equalized before applying this method.
    :param img:
    :return:
    """
    from scipy.signal import argrelextrema
    from sklearn.neighbors import KernelDensity
    import utils.histoutils as hutils

    im2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    im2 = im2[:, :, 0]
    im2 = hutils.equalize_hist(im2)
    im2 = cv2.blur(im2, (3, 3))

    X = im2.reshape([-1, 1])
    kde = KernelDensity(kernel='gaussian', bandwidth=16).fit(X)

    s = np.linspace(0, 255)
    e = kde.score_samples(s.reshape(-1, 1))
    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]

    clusters = X[X < mi[0]], X[(X >= mi[0]) * (X <= mi[1])], X[X >= mi[1]]
    pass


def apply_glare_removing(img, min_value=0.85, max_sat=0.15, convert_to_hsv=True):
    """
    the glare is present in areas of the image with LOW saturation and HIGH value,
    we mark this areas as 0 and apply an inpaint algorithm to reconstruct image
    :param img:
    :param max_sat:
    :param min_value:
    :param convert_to_hsv:
    :return:
    """
    if convert_to_hsv:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        hsv = img

    # remov
    mask = np.zeros(img.shape[0:2], np.uint8)
    mask[(hsv[:, :, 2] >= min_value * 255.) & (hsv[:, :, 1] <= max_sat * 255.)] = 255

    img = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS);
    return img


def apply_glare_removing_bn(img):
    # glare removing
    bn = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(bn, 220, 255, cv2.THRESH_BINARY);
    img3 = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS);
    return img3


if __name__ == '__main__':
    import testutils as tsu
    from utils.image.image_pipeline import make_pipeline

    # tsu.test_samples(make_pipeline([quantization_ked]))
    tsu.test_samples(make_pipeline([apply_glare_removing], store_tmp_img="composite"))
