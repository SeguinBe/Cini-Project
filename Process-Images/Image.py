import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shared
from base import DocumentInfo

class ExtractedImage:
    def __init__(self, document_info: DocumentInfo, document=None):
        self.document_info = document_info
        if document is not None:
            self._image = document
        else:
            self._image = cv2.imread(os.path.join(self.document_info.output_folder,
                                                  shared.PREDICTION_CARDBOARD_DEFAULT_FILENAME))

    def save_image(self, path=None):
        assert self._image is not None
        if path is None:
            self.document_info.check_output_folder()
            plt.imsave(os.path.join(self.document_info.output_folder, shared.PREDICTION_CARDBOARD_DEFAULT_FILENAME), self._image.astype(np.uint8))
        else:
            plt.imsave(path, self._image.astype(np.uint8))

