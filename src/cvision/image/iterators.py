import os

import cv2
import skimage.io as skio


class ImageFolderIterator(object):
    """
    Looks for images in the selected folder and interates over them
    returning numpy object representing the image
    """

    folder = None
    current_pos = 0
    gray_scale = False
    # stores folder content names
    image_file_list = []

    def __init__(self, image_folder, opener="ocv", gray_scale=False, get_file_name=False, suffix=None):
        self.suffix = suffix
        self.folder = image_folder
        self.opener = opener
        self.gray_scale = gray_scale
        self.__read_folder_content(self.folder)
        self.get_file_name = get_file_name

    def __read_folder_content(self, folder_path):
        self.image_file_list = sorted(os.listdir(folder_path))
        self.image_file_list = [os.path.join(self.folder, x) for x in self.image_file_list if
                                os.path.isfile(os.path.join(self.folder, x))]
        if self.suffix:
            self.image_file_list = [x for x in self.image_file_list if x.endswith(self.suffix)]

    def __iter__(self):
        return self

    def next(self):
        if self.current_pos >= len(self.image_file_list):
            raise StopIteration()
        else:
            img = None
            filename = self.image_file_list[self.current_pos]
            self.current_pos += 1
            if self.opener == "ocv":
                isColor = 1
                if self.gray_scale:
                    isColor = 0
                img = cv2.imread(filename, isColor)
            else:
                # skimage
                img = skio.imread(filename, as_grey=self.gray_scale)
            if img is None:
                raise ValueError("Error trying to read image: {}".format(filename))

            if self.get_file_name:
                return img, filename
            else:
                return img


if __name__ == '__main__':
    folder = "/home/gus/workspaces/wks-python/pigspy/resources/pesajes_samples"

    iterator = ImageFolderIterator(folder)

    for img in iterator:
        print img.shape
