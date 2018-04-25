import os
from scipy import misc
import matplotlib.image as mpimg
import numpy as np
import cv2
from rawkit.raw import Raw
from PIL import Image


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def load_jpg_file_to_image(filename):
    """
    Load a jpg file
    :param filename:
    :return: PIL.Image
    """
    image = misc.imread(filename)
    return image


def save_image(filename, arr: np.ndarray):
    im = Image.fromarray(arr.astype(np.uint8, copy=False))
    im.save(filename, quality=90)
    #misc.imsave(filename, arr.astype(np.uint8, copy=False))


def load_raw_file_to_image(filename):
    """
    Load a Raw file
    :param filename:
    :return: PIL.Image
    """
    with Raw(filename=filename) as raw:
        raw.options.rotation = 0
        w = raw.data.contents.sizes.width
        h = raw.data.contents.sizes.height
        buffered_image = np.array(raw.to_buffer())
        image = Image.frombytes('RGB', (w, h), buffered_image)
        return image
