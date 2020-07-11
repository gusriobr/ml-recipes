import cv2


def apply_threshold(img):
    ret, th = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    # blur = cv2.GaussianBlur(img, (5, 5), 0)
    # # ret, im1 = cv2.threshold(blur,50, 255, cv2.THRESH_BINARY)
    # im1 = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # ret, im1 = cv2.threshold(im1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def apply_glare_threshold(img):
    # if image is not binary, transform
    bn = img
    if len(img.shape)>2:
        bn = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, img2 = cv2.threshold(bn, 180, 255, cv2.THRESH_BINARY);
    # img3 = cv2.inpaint(img, img2, 3, cv2.INPAINT_TELEA);
    img3 = cv2.inpaint(img, img2, 3, cv2.INPAINT_NS);
    return img3