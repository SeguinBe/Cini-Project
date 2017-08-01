RECTO_SUBSTRING = '_recto.cr2'
RECTO_SUBSTRING_JPG = '_recto.jpg'
RECTO_MD5_SUBSTRING = '_recto.md5'
VERSO_SUBSTRING = '_verso.cr2'
VERSO_SUBSTRING_JPG = '_verso.jpg'
VERSO_MD5_SUBSTRING = '_verso.md5'
SAMPLES_DIR = 'samples'
VISITED_LOG_FILE_NAME = 'processed.txt'

PREDICTION_CARDBOARD_DEFAULT_FILENAME = "prediction.png"
EXTRACTION_THUMBNAIL_DEFAULT_FILENAME = "extraction.png"
RECTO_CARDBOARD_DEFAULT_FILENAME = 'cardboard-re.png'
VERSO_CARDBOARD_DEFAULT_FILENAME = 'cardboard-ve.png'
IMAGE_DEFAULT_FILENAME = 'image.png'
TEXT_SECTION_DEFAULT_FILENAME = 'text-section.png'

# IMAGE AND TEXT SECTION CROPPING
ACCEPTABLE_TEXT_SECTIONS_Y_RANGES = [(1100, 1390)]
RESIZE_HEIGHT = 1000.0
IMAGE_HEIGHT_LIMIT = 0.9 * RESIZE_HEIGHT
IMAGE_WIDTH_DELIMITER = 0.05 * RESIZE_HEIGHT
IMAGE_MASK_BORDER_WIDTH = 15

# CARDBOARD CROPPING
CARDBOARD_MIN_WIDTH = 4550
CARDBOARD_MAX_WIDTH = 4720

CARDBOARD_MIN_HEIGHT = 5150
CARDBOARD_MAX_HEIGHT = 5350

CARDBOARD_MIN_RATIO = 1.10
CARDBOARD_MAX_RATIO = 1.15

