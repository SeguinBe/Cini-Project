import os.path

import numpy as np
import unwarp
import utils
from Image import *
from base import *
from cardboard import *
from scipy.ndimage.interpolation import map_coordinates
from skimage.transform import warp_coords
from typing import Union



class RawScan:

    def __init__(self, document_info: DocumentInfo, base_path: str):
        """
        :param document_info:
        :param base_path: Base path of the files to be examined (ex : '/mnt/Cini/1A/1A_37')
        :return:
        """
        self.document_info = document_info
        self.cropped_cardboard = None

        if self.document_info.side == 'recto':
            self.image_path = base_path + shared.RECTO_SUBSTRING_JPG
        else:
            self.image_path = base_path + shared.VERSO_SUBSTRING_JPG

        # Checks
        assert os.path.exists(self.image_path)

        # Loads the image

        self.raw_scan = utils.load_jpg_file_to_image(self.image_path)

        self.output_prediction = shared.PREDICTION_CARDBOARD_DEFAULT_FILENAME

        if self.document_info.side == 'recto':
            self.output_filename = shared.RECTO_CARDBOARD_DEFAULT_FILENAME
        else:
            self.output_filename = shared.VERSO_CARDBOARD_DEFAULT_FILENAME

    def crop_cardboard(self, model):

        # Performs the crop
        mat = np.array(self.raw_scan.resize((1024, 688)))
        mat = mat.reshape(1, 688, 1024, 3)
        mat = mat.astype(np.uint8)


        prediction = model.gen_prediction(mat)[0]

        full_size_image = np.asarray(self.raw_scan.resize((2048, 1376)))

        self.p, self.prediction, self.angle, self.center_x, self.center_y = unwarp.get_uwrap(prediction)

        self.prediction = map_coordinates(self.prediction, warp_coords(self.transform, self.prediction.shape), order=1, prefilter=False)
        # rotate it here
        full_size_image = unwarp.rorate_image(full_size_image, self.angle)
        self.warped_image = map_coordinates(full_size_image, warp_coords(self.transform, full_size_image.shape), order=1, prefilter=False)

        rect = cv2.minAreaRect(np.argwhere(self.prediction > 0))
        self.cropped_cardboard = self.crop_minAreaRect(self.warped_image, rect)


    def crop_image(self):

        assert self.prediction is not None, "Call crop_cardboard first"

        angle = cv2.minAreaRect(np.argwhere(self.prediction > 1))[2]
        rotated_img = unwarp.rorate_image(self.warped_image, angle)
        rotated_pred = unwarp.rorate_image(self.prediction, angle)
        rect = cv2.minAreaRect(np.argwhere(rotated_pred > 1))
        self.cropped_image = self.crop_minAreaRect(rotated_img, rect)



        # Performs the checks
        # h, w = self.cropped_cardboard.shape[:2]
        # if h < w:
        #    self.cropped_cardboard = self.cropped_cardboard.transpose(1, 0, 2)[::-1]
        #    self.document_info.logger.info('Rotated the cardboard')
        #    h, w = self.cropped_cardboard.shape[:2]
        # if not self._validate_height(h):
        #    self.document_info.logger.warning('Unusual cardboard height : {}'.format(h))
        # if not self._validate_width(w):
        #    self.document_info.logger.warning('Unusual cardboard width : {}'.format(w))
        # if not self._validate_ratio(h / w):
        #    self.document_info.logger.warning('Unusual cardboard ratio : {}'.format(h / w))

    def crop_minAreaRect(self, img, rect):
        rect0 = (rect[0], rect[1], 0.0)
        box = cv2.boxPoints(rect)
        # rotate bounding bo
        pts = np.int0(box)
        img_crop = img[np.min(pts[:, 0]) + 2:np.max(pts[:, 0]) - 2, np.min(pts[:, 1]) + 2:np.max(pts[:, 1]) - 2]
        return img_crop

    def transform(self, xy):

        normalize = (np.max(xy[:, 0]) - np.min(xy[:, 0])) * (np.max(xy[:, 1]) - np.min(xy[:, 1]))
        x = xy[:, 1]
        y = xy[:, 0]
        radius = (np.square(x - self.center_x) + np.square(y - self.center_y)) / normalize
        coef_x = 1 + (radius * self.p[0][0]) + (np.square(radius) * self.p[0][1])
        coef_y = 1 + (radius * self.p[2][0]) + (np.square(radius) * self.p[2][1])

        add_x = ((self.p[1][1] * (radius + (2 * np.square(x)))) + (2 * self.p[1][0] * x * y)) * (
            1 + radius * self.p[1][2]) + (
                    np.square(radius) * self.p[1][3])
        add_y = ((self.p[3][1] * (radius + (2 * np.square(y)))) + (2 * self.p[3][0] * x * y)) * (
            1 + radius * self.p[3][2]) + (
                    np.square(radius) * self.p[3][3])

        x = (x * coef_x) + add_x
        y = (y * coef_y) + add_y
        xy = np.concatenate([y, x])

        return xy.astype(np.int32)


    def get_cardboard(self) -> Union['RectoCardboard', 'VersoCardboard']:
        assert self.cropped_cardboard is not None, 'Call crop_cardboard first'
        if self.document_info.side == 'recto':
            return RectoCardboard(self.document_info, self.cropped_cardboard)
        else:
            return VersoCardboard(self.document_info, self.cropped_cardboard)


    def get_image(self) -> 'ExtractedImage':
        assert self.cropped_image is not None, 'Call crop_image first'
        return ExtractedImage(self.document_info, self.cropped_image)


    def save_cardboard(self, path=None):
        assert self.cropped_cardboard is not None, 'Call crop_cardboard first'
        if path is None:
            self.document_info.check_output_folder()
            cv2.imwrite(os.path.join(self.document_info.output_folder, self.output_filename), self.cropped_cardboard)
        else:
            cv2.imwrite(path, self.cropped_cardboard)


    def save_prediction(self, path=None):
        assert self.prediction is not None, 'Call crop_cardboard first'
        if path is None:
            self.document_info.check_output_folder()
            plt.imsave(os.path.join(self.document_info.output_folder, self.output_prediction), self.prediction.astype(np.uint8))
        else:
            plt.imsave(path, self.prediction)

    @staticmethod
    def _validate_width(width):
        return shared.CARDBOARD_MIN_WIDTH <= width <= shared.CARDBOARD_MAX_WIDTH

    @staticmethod
    def _validate_height(height):
        return shared.CARDBOARD_MIN_HEIGHT <= height <= shared.CARDBOARD_MAX_HEIGHT

    @staticmethod
    def _validate_ratio(ratio):
        return shared.CARDBOARD_MIN_RATIO <= ratio <= shared.CARDBOARD_MAX_RATIO