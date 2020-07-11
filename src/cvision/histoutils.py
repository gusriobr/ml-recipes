import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm




def equalize_CLAHE(img):
    # create a CLAHE object (Arguments are optional).
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def equalize_hist(img, impl_type="ocv"):
    if impl_type == "ocv":
        # if image is BN
        if (len(img.shape) == 1):
            return cv2.equalizeHist(img)
        else:
            # on color images, the equalization must be done using a color space
            # that treats the intensity as a separate channel (HSV, YUB, YCbCr (preferred),
            # https://stackoverflow.com/questions/15007304/histogram-equalization-not-working-on-color-image-opencv
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            channels = cv2.split(ycrcb)
            channels[0] = cv2.equalizeHist(channels[0])
            ycrcb = cv2.merge(channels)
            img = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)

            # # using skimagen
    elif impl_type == "skimage":
        import skimage.exposure as iexp
        from skimage import img_as_float, img_as_ubyte

        imgf = img_as_float(img)
        img = iexp.equalize_adapthist(imgf)
        img = img_as_ubyte(img)
    else:
        raise Exception("Incorrect implementation 'ocv' or 'skimage' expected: {}".format(impl_type))

    return img


def cielab_equalization(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = equalize_hist(img)
    return img


def image_histogram(image):
    # http://docs.opencv.org/3.3.0/d1/db7/tutorial_py_histogram_begins.html
    hist, bins = np.histogram(image.ravel(), bins=256, range=[0, 256])
    return hist


def hsv_2dhistograms(hsv):
    # convertir a hsv
    fig, axes = plt.subplots(1, 3, squeeze=False)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    plt.subplot(131)
    plt.imshow(hist, interpolation='nearest')
    plt.ylim(0, 180)
    plt.ylabel("H")
    plt.xlabel("S")

    hist = cv2.calcHist([hsv], [0, 2], None, [180, 256], [0, 180, 0, 256])
    plt.subplot(132)
    plt.imshow(hist, interpolation='nearest')
    plt.ylim(0, 180)
    plt.ylabel("H")
    plt.xlabel("V")

    hist = cv2.calcHist([hsv], [1, 2], None, [256, 256], [0, 256, 0, 256])
    plt.subplot(133)
    plt.imshow(hist, interpolation='nearest')
    plt.ylim(0, 256)
    plt.ylabel("S")
    plt.xlabel("V")

    plt.show()


def histogram_1d(img, channel=0, size=180):
    plt.figure()
    mask = img
    histr = cv2.calcHist([img], [channel], mask, [size], [0, size])
    plt.plot(histr, color="g")
    plt.xlim([0, size])


def histogram_2d(img, channels=[], sizes=[], titles=[]):
    # convertir a hsv
    hist = cv2.calcHist([img], channels, None, [sizes[0], sizes[1]], [0, sizes[1], 0, sizes[1]])
    plt.figure()
    plt.imshow(hist, interpolation='nearest')
    plt.ylabel(titles[0])
    plt.xlabel(titles[1])


def hsl_2dhistogram(img):
    # convertir a hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    plt.figure()
    plt.imshow(hist, interpolation='nearest')
    plt.ylabel("H")
    plt.xlabel("S")


def cielab_2dhistogram(img, convert_cie=True, title = ""):
    cielab = img
    if convert_cie:
        cielab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    hist = cv2.calcHist([cielab], [1, 2], None, [150, 150], [0, 150, 0, 150])
    plt.figure()
    plt.imshow(hist, interpolation='nearest')
    plt.ylabel("A")
    plt.xlabel("B")
    plt.title(title)


def cielab_2dhistogram_v2(img, convert_cie=True, title = ""):
    cielab = img
    if convert_cie:
        cielab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(cielab)
    nbins = 20
    plt.hist2d(a.flatten(), b.flatten(), bins=nbins, norm=LogNorm())
    plt.xlabel('A')
    plt.ylabel('B')
    plt.xlim([40, 230])
    plt.ylim([20, 230])
    plt.title(title)

def hsv_2dhistogram_v2(img, convert_hsv=False, title = ""):
    img_color = img
    if convert_hsv:
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(img_color)
    nbins = 20
    plt.hist2d(h.flatten(), s.flatten(), bins=nbins, norm=LogNorm())
    plt.xlabel('Hue')
    plt.ylabel('Saturation')
    plt.xlim([0, 180])
    plt.ylim([0, 255])
    plt.title(title)

def hsl_2dhistogram_v3(img, convert_hsv = False):
    x = img[:, :, 0].reshape((-1))  # hue
    y = img[:, :, 1].reshape((-1))  # saturation
    x = x.T
    y = y.T

    # Estimate the 2D histogram
    nbins = 120
    H, xedges, yedges = np.histogram2d(x, y, bins=(180, 256))
    H = H.T

    # # Mask zeros
    Hmasked = np.ma.masked_where(H == 0, H)  # Mask pixels with a value of zero
    #
    # # Plot 2D histogram using pcolor
    X, Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X, Y, Hmasked)
    plt.xlabel('Hue')
    plt.ylabel('Saturation')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')


def rgb_histogram(img):
    color = ('b', 'g', 'r')
    plt.figure()
    for channel, col in enumerate(color):
        histr = cv2.calcHist([img], [channel], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.title('RGB color histogram')


def calculate_lbp_histogram(lbp):
    n_bins = int(lbp.max() + 1)
    (hist, _) = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

    # normalize the histogram
    hist = hist.astype("float32")
    hist /= hist.sum()
    return hist


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def plot_histograms(histograms_map, ncols=3):
    total = len(histograms_map)
    nrows = int(total / ncols) + (total % ncols)

    (1 if total % ncols == 0 else 1)

    plt.figure(1)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False)
    keys = histograms_map.keys();
    for row in range(nrows):
        for col in range(ncols):
            pos = row * ncols + col
            if pos < len(keys):
                k = keys[pos]
                hist = histograms_map[k]
                N = len(hist)
                x = range(N)
                width = 1 / 1.5
                axes[row][col].bar(x, hist, width)
                # axes[row][col].axis('off')
                axes[row][col].set_title("{}".format(k))
                # plt.show()


def plot_image_hist(img, channels=[0, 1, 2], ranges=[[0, 256], [0, 256], [0, 256]]):
    """
    Print histogram for separated RGB channels
    :param img:
    :param channels:
    :return:
    """
    color = ('b', 'g', 'r', 'y')
    plt.figure()
    for c in channels:
        rng = ranges[c]
        histr = cv2.calcHist([img], [c], None, [rng[1]], [rng[0], rng[1]])
        plt.plot(histr, color=color[c])
        plt.xlim(rng)


def distance_function(h1, h2, f_distance="CHISQR"):
    if f_distance.lower() == "chisqr":
        dist = cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR)
    elif f_distance.lower() == "kullback_leibler":
        dist = kullback_leibler_divergence(h1, h2)
    elif f_distance.lower() == "correlation":
        dist = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        if (dist == 0):
            dist = 999999;
        dist = 1 / dist
    elif f_distance.lower() == "hellinger":
        dist = cv2.compareHist(h1, h2, cv2.HISTCMP_HELLINGER)
    elif f_distance.lower() == "intersect":
        dist = cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)
        if (dist == 0):
            dist = 999999;
        dist = 1 / dist

    return dist


if __name__ == '__main__':
    img = cv2.imread("C:/Desarrollo/workspaces/wks-py/pigsty/tmp/samples/frame360.jpg")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv_2dhistogram(hsv)
    hsl_2dhistogram_v2(hsv)
    plt.show()

    # img = cv2.imread("/tmp/prueba_2.png")
    # hsv_2dhistogram(img)
    #
    # hsv[(hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 200)] = 0
    # hsv_2dhistogram(hsv)
    #
    # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.imshow("asdf", rgb);
    # cv2.waitKey(0)
    #
    #
    #
    #
