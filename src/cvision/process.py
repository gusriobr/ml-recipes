import multiprocessing as mp
import os
from itertools import izip_longest

import pigsty.config as cfg
import utils.video.edition as vedit
import utils.video.video_pipeline as vdp
from utils.image.image_pipeline import make_pipeline

"""
  PARALLEL TASK PROCESSING FUNCTIONS
"""


class VideoProcessorWorker(object):
    v_list = None

    def __init__(self, video_list, processing_func=None, make_ppl_func=None, out_fname_func=None):
        """
        :param processing_func: function used to processed each video frame, a python function or a
        FramePipeLine can be used
        :param video_list:
        """
        # image processing function
        self.img_proc_func = processing_func
        self.make_ppl_func = make_ppl_func
        self.out_fname_func = out_fname_func

        self.v_list = []
        if type(video_list) == list:
            self.v_list.extend(video_list)
        else:
            self.v_list.append(video_list)

        for v in self.v_list:
            if not os.path.isfile(v):
                raise Exception("Video file does not exists: " + v)

    def process_video_file(self, video_file):

        tag = "VPROC"
        cfg.setTaskTag(tag)
        ppl = None

        if self.make_ppl_func:
            ppl = self.make_ppl_func(tag, video_file)
        else:
            ppl = make_pipeline(steps=[(self.img_proc_func, tag, video_file)])

        output_file = None
        if self.out_fname_func:
            # calculate output filename
            output_file = self.out_fname_func(video_file)

        print (">>>> Processing video: {}".format(video_file))
        if output_file:
            print ("Output video file: {}".format(output_file))

        v_path = vdp.run_video_ppl(video_file, ppl, output_video=output_file)
        print (">>>> Video processing finished")

        return [v_path]

    def process_video(self, tag):
        if not tag:
            return None

        print (">>>> Executing task TAG " + tag)

        cfg.setTaskTag(tag)
        ppl = None

        vpath_list_ret = []
        for video_path in self.v_list:
            if self.make_ppl_func:
                ppl = self.make_ppl_func(tag, video_path)
            else:
                ppl = make_pipeline(steps=[(self.img_proc_func, tag, video_path)])

            output_file = None
            if self.out_fname_func:
                # calculate output filename
                output_file = self.out_fname_func(video_path)

            v_path = vdp.run_video_ppl(video_path, ppl, output_video=output_file)
            vpath_list_ret.append(v_path)

        print (">>>> Task Finished TAG " + tag)

        return vpath_list_ret


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)


def parallel(video_list, impl_func=None, make_ppl_func=None, num_parallel_procs=None, combine_video=False,
             combined_out_prefix=None, num_columns=2, out_fname_func=None):
    nproc = _check_num_proc(num_parallel_procs, video_list)

    _check_impl_func(impl_func, make_ppl_func)

    v_path_list = []  # returning video list
    v_idx = 0
    for selected_videos in grouper(video_list, nproc, fillvalue=None):
        # convert tuple to list
        vlist = [x for x in selected_videos if x is not None]
        pool = mp.Pool(processes=len(vlist))
        v_processor = VideoProcessorWorker(vlist, processing_func=impl_func, make_ppl_func=make_ppl_func,
                                           out_fname_func=out_fname_func)
        results = pool.map(v_processor.process_video_file, vlist)
        pool.close()
        pool.join()

        for result in results:
            if result is not None:
                v_path_list.append(result[0])
        v_idx += len(selected_videos)

        print (">>>[Finished {} of {} tasks]".format(v_idx, len(video_list)))

    if combine_video and len(video_list) > 1:
        # sort video_paths to make sure video images are always in the same order
        v_path_list = sorted(v_path_list)
        prefix = combined_out_prefix
        if not prefix:
            prefix = "COMB_" + "__".join(video_list) + " "

        vedit.compose(video_paths=v_path_list, cols=num_columns, out_prefix=prefix)

def parallel_task(tags, video_list, impl_func=None, make_ppl_func=None, num_parallel_procs=None, combine_video=True,
                  combined_out_prefix=None, num_columns=2, out_fname_func=None):
    nproc = _check_num_proc(num_parallel_procs, video_list)

    _check_impl_func(impl_func, make_ppl_func)

    for v_idx, v_path in enumerate(video_list):
        v_path_list = []  # returning video list
        for selected_tags in grouper(tags, nproc, fillvalue=None):
            tag_list = []
            tag_list.extend(selected_tags)

            pool = mp.Pool(processes=len(tag_list))

            v_processor = VideoProcessorWorker(v_path, processing_func=impl_func, make_ppl_func=make_ppl_func,
                                               out_fname_func=out_fname_func)
            results = pool.map(v_processor.process_video, tag_list)
            pool.close()
            pool.join()

            for result in results:
                if result is not None:
                    v_path_list.append(result[0])

        print (">>>[Finished {} of {} tasks]".format(v_idx + 1, len(video_list)))

        if combine_video and len(tags) > 1:
            # sort video_paths to make sure video images are always in the same order
            v_path_list = sorted(v_path_list)
            prefix = combined_out_prefix
            if not prefix:
                prefix = "COMB_" + "__".join(tags) + " "

            vedit.compose(video_paths=v_path_list, cols=num_columns, out_prefix=prefix)


def _check_impl_func(impl_func, make_ppl_func):
    if not impl_func and not make_ppl_func:
        raise Exception(
            "One of this parameters must be set to define the image processing function: impl_func, make_ppl_func")
    if impl_func and make_ppl_func:
        raise Exception(
            "JUST one of this parameters must be set to define the image processing function: impl_func, make_ppl_func")


def _check_num_proc(num_parallel_procs, video_list):
    if not num_parallel_procs:
        return mp.cpu_count() - 1
    if num_parallel_procs > len(video_list):
        return len(video_list)

    return num_parallel_procs
