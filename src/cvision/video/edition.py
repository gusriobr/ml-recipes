import os

import cv2
import numpy as np

import pigsty.config as cfg
import ocvtools.image.imutils as imu
from cvision.video.iterators import VideoFrameIterator
from cvision.video.writter import VideoWritter


def is_active_iterator(video_iterators):
    """
    Check if there's still one iterator with pending frames to process
    :param video_iterators:
    :return:
    """
    for key, vit in video_iterators.iteritems():
        if vit is not None:
            return True
    return False


def __add_file_name(img, path):
    file_name = os.path.basename(path)
    position = (10, img.shape[0] - 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    if len(img.shape) < 3:  # bn image
        color = (255)
    img2 = cv2.putText(img, file_name, position, font, 1, color, 1, cv2.LINE_AA)
    return img2


def compose(video_paths=None, folder_path=None, out_prefix=None, cols=1, add_text=True):
    """
        combine multiple videos using the images prov

    :param video_paths: list of video paths to include in the video
    :param folder_path: folder where video can be found. All video files (.avi or .mpeg will be included
    :param cols:
    :param add_text:
    :return:
    """

    if folder_path:
        # get all video in the folder.
        lst = os.listdir(folder_path)
        lst = [os.path.join(folder_path, x) for x in lst]
        # keep files
        lst = [x for x in lst if os.path.isfile(x)]
        # keep video files
        lst = [x for x in lst if os.path.splitext(x)[1].lower() == ".avi"
               or os.path.splitext(x)[1].lower() == ".mpeg"]

        video_paths = [os.path.join(folder_path, x) for x in lst]

    video_iterators = {}

    for path in video_paths:
        if not os.path.isfile(path):
            raise Exception("The file {} does not exists!!!".format(path))
        video_iterators[path] = VideoFrameIterator(path)

    img_pile = []
    img_shape = None
    vwritter = None
    while is_active_iterator(video_iterators):
        for key, vit in video_iterators.iteritems():
            if vit is None:  # iterator has already finished
                img = np.zeros(img_shape, np.uint8)
            else:
                try:
                    img = vit.next()
                except StopIteration:
                    # iteration over this file is over
                    img = None
                    video_iterators[key] = None

            if img is None:
                # mark iterator as finished
                video_iterators[key] = None
                img = np.zeros(img_shape, np.uint8)

            if img_shape is None:
                img_shape = img.shape

            if add_text:
                __add_file_name(img, key)

            img_pile.append(img)

        img_proc = imu.compose(img_pile, cols=cols)
        del img_pile[:]

        # write processed video_frame
        if not vwritter:
            file_name = cfg.getTempVideoFileName(prefix=out_prefix)
            print (file_name)
            vwritter = VideoWritter(file_name)

        vwritter.add_img(img_proc)

    vwritter.close()


if __name__ == '__main__':
    # v_list = ['C:/Desarrollo/workspaces/wks-py/pigsty/resources/pesajes/video2_roi.avi',
    #           'C:/Desarrollo/workspaces/wks-py/pigsty/resources/pesajes/video1_roi.avi']

    v_list = ['C:/Desarrollo/workspaces/wks-py/pigsty/resources/pesajes/video2_roi.avi',
              'C:/Desarrollo/workspaces/wks-py/pigsty/tmp/videos/hsv/FELZEN_HSV.avi']

    # video_folder = "C:/Desarrollo/workspaces/wks-py/pigsty/tmp/videos/hsv"
    # v_list = [os.path.join(video_folder, f) for f in os.listdir(video_folder)]

    cfg.setTaskTag("HSV_COMP")
    compose(v_list, cols=1)
