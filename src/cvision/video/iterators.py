import os

import cv2

import info as vinfo


class VideoFrameIterator(object):
    video_file = None
    start_second = None
    end_second = None
    frame_range = None
    # iteration info
    opened = False
    num_frames = None
    total_frames = None
    vid_info = None
    vid_cap = None

    def __init__(self, video_file, seconds_range=[0, -1], frame_range=None):
        self.video_file = video_file
        self.start_second = seconds_range[0]
        self.end_second = seconds_range[1]
        self.num_frames = 0
        self.total_frames = 0
        self.frame_range = frame_range

    def __iter__(self):
        return self

    def __open_video(self, video_file):
        self.vidcap = cv2.VideoCapture(self.video_file)
        success, frame = self.vidcap.read()
        self.total_frames += 1

        if self.__invalid_image(frame):
            self.vidcap.release()
            raise ValueError("First frame couldn't be processes to get fps info.")
        if not success:
            self.vidcap.release()
            raise StopIteration()

        self.vid_info = vinfo.get_video_info(self.video_file)
        self.opened = True
        return success, frame

    def next(self):
        # open video file
        success = frame = None
        if not self.opened:
            success, frame = self.__open_video(self.video_file)
            return frame

        while True:
            if not self.frame_range:
                # filter frames using second range
                frame = self.__next_by_seconds()
            else:
                frame = self.__next_by_nframes()

            if self.__apply_current_frame():
                break;

        return frame

    def __next_by_seconds(self):
        success, frame = self.vidcap.read()
        self.total_frames += 1

        # skip initial seconds as dictated by "start_second" param
        current_second = self.total_frames / self.vid_info.fps
        while success and current_second <= self.start_second:
            success, frame = self.vidcap.read()
            self.total_frames += 1
            current_second = self.total_frames / self.vid_info.fps

        if not success \
                or (current_second >= self.end_second and self.end_second != -1):
            self.vidcap.release()
            raise StopIteration()

        if self.__invalid_image(frame):
            self.vidcap.release()
            raise ValueError("Error reading frame number {}.".format(self.total_frames))

        return frame

    def __next_by_nframes(self):
        success, frame = self.vidcap.read()
        self.total_frames += 1

        # skip initial frames as dictated by "frame_range" init param
        while success and self.total_frames < self.frame_range[0]:
            success, frame = self.vidcap.read()
            self.total_frames += 1

        if not success or self.total_frames > self.frame_range[1]:
            self.vidcap.release()
            raise StopIteration()

        if self.__invalid_image(frame):
            self.vidcap.release()
            raise ValueError("Error reading frame number {}.".format(self.total_frames))

        return frame

    def __invalid_image(self, frame):
        return frame is None

    def __process_current_frame(self, frame):
        return

    def __apply_current_frame(self):
        """
        Extension point for subclasses
        :param frame:
        :return:
        """
        return True

    def get_video_info(self):
        return self.vid_info




if __name__ == '__main__':
    video_path = "/home/gus/workspaces/wks-python/pigspy/resources/v_canny.avi"

    viterator = VideoFrameIterator(video_path)  # , frame_range=[0,10])
    value = 0;
    for img in viterator:
        value += 1
        print img.shape

    print value

    viterator = VideoFrameIterator(video_path, frame_range=[0, 10])
    value = 0;
    for img in viterator:
        value += 1
        print img.shape

    print value
