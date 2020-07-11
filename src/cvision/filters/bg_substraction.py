import cv2
import numpy as np

from pigsty.segmentation.contours import get_contour_mask


def apply_graph_cut(img):

    contour_mask = get_contour_mask(img, 5600)

    if contour_mask is not None:
        x,y,w,h = cv2.boundingRect(contour_mask)

        # graph cut
        mask = np.zeros(img.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = x,y,w,h

        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
    return img


def apply_bg_substraction(img, fgbg, kernel):
    fgmask = fgbg.apply(img)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    ret = cv2.bitwise_and(img, img, mask=fgmask)

    return ret


def apply_mog_filter(frame, fgbg, kernel, applyMask):
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(frame)

    th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (3, 3)), iterations=2)
    image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 1600:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    return frame


def apply_mog_on_frame(img, params):
    fgbg = params[0]
    kernel = params[1]
    apply_mask = False
    if len(params) > 2:
        apply_mask = True

    frame = img
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # learning rate =0 --> background model is fixed, no high-frequency elements
    fgmask = fgbg.apply(frame, learningRate=0)
    fgmask = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]

    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=5)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=3)

    image, contours, hier = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 1600:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    # fgmask = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3)), iterations=2)

    ret = fgmask
    if apply_mask:
        # return image after applying obtained mask
        ret = cv2.bitwise_and(img, img, mask=fgmask)

    return ret
