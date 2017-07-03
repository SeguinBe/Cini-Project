import os.path

import numpy as np
import unwarp
import utils
from image import ExtractedImage
from base import *
from cardboard import RectoCardboard, VersoCardboard
import shared
import cv2
import matplotlib.pyplot as plt
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

    def crop_cardboard(self, model, do_unwarp=False, crop_image=True):
        # Performs the crop
        target_h, target_w = (688, 1024)
        full_size_image = np.asarray(self.raw_scan)
        original_h, original_w = full_size_image.shape[:2]
        mat = cv2.resize(full_size_image, (target_w, target_h))

        prediction = model.predict(mat[None, :, :, :])[0]
        # Switch classes
        prediction[prediction == 0] = 3
        prediction[prediction == 1] = 0
        prediction[prediction == 3] = 1
        self.prediction = prediction
        self.prediction_scale = target_h/original_h

        cardboard_prediction = unwarp.get_cleaned_prediction(prediction > 0)
        _, contours, hierarchy = cv2.findContours(cardboard_prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cardboard_contour = contours[np.argmax([cv2.contourArea(c) for c in contours])]
        if do_unwarp:
            self.p, self.center_x, self.center_y = unwarp.uwrap(self.prediction)
            self.prediction = map_coordinates(self.prediction, warp_coords(self.transform, self.prediction.shape),
                                              order=1, prefilter=False)
            self.warped_image = map_coordinates(full_size_image, warp_coords(self.transform, full_size_image.shape),
                                                order=1, prefilter=False)
        else:
            self.warped_image = full_size_image

        self.cropped_cardboard = self.extract_minAreaRect(self.warped_image, cv2.minAreaRect(cardboard_contour),
                                                          scale=1/self.prediction_scale)

        if crop_image:
            image_prediction = unwarp.get_cleaned_prediction(self.prediction > 1)
            _, contours, hierarchy = cv2.findContours(image_prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            rect = cv2.minAreaRect(np.concatenate(contours))
            self.cropped_image = self.extract_minAreaRect(self.warped_image, rect,
                                                          scale=1/self.prediction_scale)

        h, w = self.cropped_cardboard.shape[:2]
        if h < w:
            self.cropped_cardboard = self.cropped_cardboard.transpose(1, 0, 2)[::-1]
            if crop_image:
                self.cropped_image = self.cropped_image.transpose(1, 0, 2)[::-1]
            self.document_info.logger.info('Rotated the cardboard')

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

    def extract_minAreaRect(self, img, rect, scale):
        center, size, angle = rect
        # Find the closest angle to vertical
        while angle > 45:
            angle -= 90
            size = (size[1], size[0])
        while angle < -45:
            angle += 90
            size = (size[1], size[0])
        # Multiply sizes by the scale factor
        center = (center[0] * scale, center[1] * scale)
        size = (size[0] * scale, size[1] * scale)

        # Generates the transformation matrix
        T = np.array([[0, 0, center[0] - size[0] / 2], [0, 0, center[1] - size[1] / 2]])
        M = cv2.getRotationMatrix2D(center, angle, 1.0) - T
        # Perform the transformation
        return cv2.warpAffine(img, M, (round(size[0]), round(size[1])))

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

    def save_prediction(self, path=None):
        assert self.prediction is not None, 'Call crop_cardboard first'
        if path is None:
            self.document_info.check_output_folder()
            path = os.path.join(self.document_info.output_folder, self.output_prediction)
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
