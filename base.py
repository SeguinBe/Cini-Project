import logging
import os.path


class DocumentInfo:
    class LoggerAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            return '%s : %s' % (self.extra['description'], msg), kwargs

    def __init__(self, basename, output_folder=None, side='recto'):
        """

        :param basename:
        :param output_folder:
        :param side:
        :return:
        """
        assert side in ['recto', 'verso']

        self.side = side
        self.output_folder = output_folder
        self.basename = os.path.basename(basename)

        # Logger
        logger = logging.getLogger(__name__)
        self.logger = DocumentInfo.LoggerAdapter(logger, {
            'description': self.basename + ('.re' if self.side == 'recto' else '.ve')
        })

    def check_output_folder(self):
        assert self.output_folder is not None, "DocumentInfo output_folder is None"
        os.makedirs(self.output_folder, exist_ok=True)

    def validate_barcode(self, detected_barcode: str):
        stripped_barcode_1 = self.basename.replace(" ", "").replace("_", "")
        stripped_barcode_2 = detected_barcode.replace(" ", "").replace("_", "")
        return stripped_barcode_1 == stripped_barcode_2
