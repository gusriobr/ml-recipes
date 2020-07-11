import cv2

class VideoInfo():
    fps = None
    resolution = (0, 0)
    codec = None
    fourcc = None
    isColor = None


def get_video_info(video_file):
    video_info = VideoInfo()

    vidcap = cv2.VideoCapture(video_file)
    success, image = vidcap.read()
    video_info.fps = vidcap.get(cv2.CAP_PROP_FPS)
    height, width, layers = image.shape
    video_info.resolution = (width, height)
    video_info.fourcc = vidcap.get(cv2.CAP_PROP_FOURCC)
    video_info.isColor = (layers is not None)

    return video_info