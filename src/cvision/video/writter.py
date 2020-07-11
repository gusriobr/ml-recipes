import cv2
import sys

class VideoWritter:
    """
    Helper to create video from images
    """
    fps = None
    resolution = None
    codec = None
    fourcc = None
    isColor = False
    video = None

    def __init__(self, out_file, resolution=(640, 480), fps=25, codec='XVID', vinfo=None):
        if not vinfo:
            self.fps = fps
            self.resolution = resolution
            self.codec = codec
            self.fourcc = cv2.VideoWriter_fourcc(*self.codec)
        else:
            self.fps = vinfo.fps
            self.resolution = vinfo.resolution
            self.fourcc = vinfo.fourcc
            self.isColor = vinfo.isColor
        if resolution:
            self.resolution = resolution
        self.out_file = out_file

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    def __detect_color(self, img):
        """
        Detects color for image and configures video writter
        :param img:
        :return:
        """
        # detect color
        self.isColor = (len(img.shape) > 2)
        color_flag = 0 if not self.isColor else 1
        resolution = self.__get_res_from_frame(img)
        # configure video writter
        self.video = cv2.VideoWriter(self.out_file, int(self.fourcc), self.fps, resolution, color_flag)

    def __get_res_from_frame(self, img):
        height, width = img.shape[:2]
        return (width, height)

    def add_img(self, img):
        if not self.video:
            self.__detect_color(img)
        self.video.write(img)

    def close(self):
        if self.video:
            self.video.release()