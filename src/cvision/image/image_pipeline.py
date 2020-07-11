import itertools
import os
import sys
import time
import cv2

import pigsty.config as cfg
from utils.image.iterators import ImageFolderIterator


def make_pipeline(steps=[], tmp_dir=None, debug=False, store_tmp_img=None):
    pipeline = FramePipeLine(tmp_dir=tmp_dir, store_tmp_img=store_tmp_img)
    for s in steps:
        if type(s) != tuple and not callable(s):
            raise Exception("The parameter {} is not a tuple nor a function.")
        # expand tuple into parameters
        if type(s) == tuple:
            pipeline.add_step(*s)
        else:
            pipeline.add_step(s)
    pipeline.debug = debug
    return pipeline


class FramePipeLine(object):
    step_functions = None
    step_parameters = None
    current_image = None
    has_variations = False
    max_num_variations = 1
    variations_grid = {}
    variations_names = []
    current_selection = None
    tmp_dir = None
    debug = False
    process_id = None
    store_tmp_img = None
    """   if temporal images must stored during frame processing. None: no images stored. indidual: one file per image.
    composed: one file with all images.
    
    """

    def __init__(self, tmp_dir=None, debug=False, store_tmp_img=None):
        self.step_functions = []
        self.step_parameters = []
        self.current_frame = 0
        self.debug = debug
        self.process_id = None
        if not tmp_dir:
            # get default OS tmp file
            tmp_dir = cfg.getDefaultTmpFolder()

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        self.tmp_dir = tmp_dir
        self.store_tmp_img = store_tmp_img
        self.__check_store_values()

    def __save_tmp_image(self, step, img):
        if not self.tmp_dir:
            raise ValueError("The tmp_dir parameter must be set")

        img_file = os.path.join(self.tmp_dir, "tmp_{}_{}.png".format(self.process_id, step))
        cv2.imwrite(img_file, img)
        return

    def __save_composite_image(self, img_list):
        """
        Creates composite image for the image list provided and stores it in the configured tmp_dir.
        :param current_img_stack:
        :return:
        """
        import ocvtools.image.imutils as imu
        if not self.tmp_dir:
            raise ValueError("The tmp_dir parameter must be set")

        composed_img = imu.compose(img_list)
        img_file = os.path.join(self.tmp_dir, "tmp_{}.png".format(self.process_id))
        cv2.imwrite(img_file, composed_img)
        return

    def add_variations(self, names, lst):
        """
        lst array of arrays, each item has a function in
        first position and the arguments to execute call in the remaining items
        :param lst:
        :return:
        """
        self.variations_names = names
        self.step_functions.append([items[0] for items in lst])
        self.step_parameters.append([items[1:] for items in lst])
        if len(lst) > self.max_num_variations:
            max_num_variations = len(lst)
        # keep variation position and number of variations in function/parameters
        self.variations_grid[len(self.step_functions) - 1] = range(len(lst))

    def get_combinations(self):
        if not self.has_variations:
            return 1
        else:
            return len(list(itertools.product(*self.variations_grid.values())))

    def add_step(self, proc_function, *args, **kwargs):
        self.step_functions.append(proc_function)
        self.step_parameters.append([args, kwargs])

    def process_variations(self, frame, output_folder=None):
        # calculate number of variations
        images = []
        for current in list(itertools.product(*self.variations_grid.values())):
            self.current_selection = current
            img = self.process_frame(frame)
            images.append(img)
            if output_folder:
                img_name = "test_{}.png".format("_".join(str(x) for x in current))
                img_file = os.path.join(output_folder, img_name)
                cv2.imwrite(img_file, img)
        return images

    def process_frame(self, frame):
        img = frame.copy()
        variation_position = 0
        step = 0
        current_img_stack = []
        if self.debug:
            print ("___________________ Frame {} ___________________".format(self.current_frame))
        self.__calc_process_id()
        for func, params in zip(self.step_functions, self.step_parameters):
            if type(func) is list:
                # get current selection to decide which function must be ran
                variation = self.current_selection[variation_position]
                f = func[variation]
                args = params[variation]
                img = f(img, *args)
                if self.debug:
                    self.__save_tmp_image(step, img)
                self.current_image = img
                variation_position += 1
            else:
                if self.debug:
                    print ("applying function {}: {}.".format(step, func.__name__))
                args = params[0]
                kwargs = params[1]
                try:
                    img = func(img, *args, **kwargs)
                except:
                    print("Unexpected error executing function {}:{}".format(func, sys.exc_info()))
                    raise

                if self.store_tmp_img == "individual":
                    self.__save_tmp_image(step, img)
                self.current_image = img
            step += 1
            current_img_stack.append(img.copy())  # store a copy to create composite image
        self.current_frame += 1
        if self.store_tmp_img == "composite":
            current_img_stack = self.tag_images(current_img_stack, zip(self.step_functions, self.step_parameters))
            self.__save_composite_image(current_img_stack)
        return img

    def __calc_process_id(self):
        """
        Calculate unique id for all temporal images
        :return:
        """
        self.process_id = hex(int(time.time() * 10))[2:].upper()

    def __check_store_values(self):
        """
        Checks possible values for store_tmp_img property
        :return:
        """
        if not self.store_tmp_img:
            return
        if self.store_tmp_img != "individual" and self.store_tmp_img != "composite":
            raise ValueError(
                "Invalid parameter value for store_tmp_img, online 'individual' and 'composite' values are recognized.")

    def tag_images(self, img_stack, paramters):
        return img_stack




def run_ppl_on_folder(img_folder, pipeline, output_video=None, debug=False, out_fname_func=None):

    img_iterator = ImageFolderIterator(img_folder, get_file_name=True)

    for img, orig_filename in img_iterator:
        im = pipeline.process_frame(img)
        # store image in test folder
        if out_fname_func is None:
            file_name = cfg.getTempImgFileName()
        else:
            file_name = out_fname_func(orig_filename)
        cv2.imwrite(file_name, im)

