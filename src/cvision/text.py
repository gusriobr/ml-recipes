import cv2


def add_text_top(img, text, font=None, font_scale=0.5, color=None, margin=[10, 10]):
    # color?
    is_color = len(img.shape) > 2

    if not font:
        font = cv2.FONT_HERSHEY_SIMPLEX

    if not color:
        color = (255, 255, 255)
        if not is_color:
            color = (255)

    if type(margin) != list and type(margin) != tuple:
        margin = (margin, margin)
    pos = (margin[0], margin[1])

    img = cv2.putText(img, text, pos, font, font_scale, color)
    return img


def add_text_bottom(img, text, font=None, font_scale=0.5, color=None, margin=[10, 10]):
    # color?
    is_color = len(img.shape) > 2

    if not font:
        font = cv2.FONT_HERSHEY_SIMPLEX

    if not color:
        color = (255, 255, 255)
        if not is_color:
            color = (255)

    if type(margin) != list and type(margin) != tuple:
        margin = (margin, margin)
    pos = (margin[0], img.shape[0] - 2 * margin[1])

    img = cv2.putText(img, text, pos, font, font_scale, color)
    return img
