import os.path

import cv2
from PIL import Image
import numpy as np
import shared
from base import DocumentInfo


class RectoCardboard:
    def __init__(self, document_info: DocumentInfo, document=None):
        self.document_info = document_info
        if document is not None:
            self.cardboard = document
        else:
            self.cardboard = cv2.imread(os.path.join(self.document_info.output_folder,
                                                     shared.RECTO_CARDBOARD_DEFAULT_FILENAME))
        self._image = None
        self._image_bounds = None
        self._text_section = None

    def save_image(self, path=None):
        assert self.cardboard is not None
        if path is None:
            self.document_info.check_output_folder()
            path = os.path.join(self.document_info.output_folder, shared.IMAGE_DEFAULT_FILENAME)
        im = Image.fromarray(self.cardboard.astype(np.uint8))
        im.save(path)

    @staticmethod
    def _validate_image_section(x, page_width):
        page_mid_x = page_width / 2
        acceptable_x_min = page_mid_x - 0.06 * page_width
        acceptable_x_max = page_mid_x + 0.06 * page_width
        return acceptable_x_min <= x <= acceptable_x_max

    @staticmethod
    def _validate_text_section(y_value):
        valid = False
        for (mini, maxi) in shared.ACCEPTABLE_TEXT_SECTIONS_Y_RANGES:
            if mini <= y_value <= maxi:
                valid = True
                break
        return valid


class VersoCardboard:
    def __init__(self, document_info: DocumentInfo, image=None):
        self.document_info = document_info
        if image is not None:
            self.cardboard = image
        else:
            self.cardboard = cv2.imread(os.path.join(self.document_info.output_folder,
                                                     shared.VERSO_CARDBOARD_DEFAULT_FILENAME))
        self.barcode = None



