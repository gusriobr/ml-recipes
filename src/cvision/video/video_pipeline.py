# simplifies video pipeline creation
import os
import tempfile
import time

from utils.image.image_pipeline import FramePipeLine
from utils.video.iterators import VideoFrameIterator
from ocvtools.video.vidutils import process_frames
import pigsty.config as cfg

def make_run_video_ppl(video_path, steps, output_video=None, tmp_dir=None, debug=False):
    """
    Simplifies video pipelines creation and running
    :param video_path:
    :param steps:
    :param output_video:
    :param tmp_dir:
    :param debug:
    :return:
    """
    if type(steps) is not list:
        raise Exception("steps param must be a list containing image processing atomic functions.")

    if not tmp_dir:
        # get default OS tmp file
        tmp_dir = tempfile.gettempdir()

    pipeline = __create_ppl_with_steps(steps, tmp_dir)
    pipeline.debug = debug

    if not output_video:
        time_stamp = (str(int(time.time())))
        output_video = os.path.join(cfg.get_default_video_folder(), time_stamp + ".avi")

    video_iterator = VideoFrameIterator(video_path)
    process_frames(video_iterator, pipeline, output_video=output_video, show_video=debug)


def __create_ppl_with_steps(steps, tmp_dir):
    pipeline = FramePipeLine(tmp_dir=tmp_dir)
    for s in steps:
        if type(s) != tuple and type(s) != function:
            raise Exception("The parameter {} is not a tuple nor a function.")
        # expand tuple into parameters
        if type(s) == tuple:
            pipeline.add_step(*s)
        else:
            pipeline.add_step(s)
    return pipeline




def run_video_ppl(video_path, pipeline, output_video=None, debug=False, num_processes=1):
    if not output_video:
        output_video = cfg.getTempVideoFileName()

    video_iterator = VideoFrameIterator(video_path)
    v_data = process_frames(video_iterator, pipeline, output_video=output_video, show_video=debug, num_processes=num_processes)
    return v_data["file_name"]
