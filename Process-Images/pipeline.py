import argparse
import glob
import os.path
import re
from pathlib import Path

from RawScan import *
from base import *
from load_model import Model
from shared import *

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--raws", required=True, help="Folder with raw images or text files with filenames to be processed")
ap.add_argument("-d", "--destination", required=True, help="Folder where the results will be saved")
ap.add_argument("-m", "--model", required=True, help="The model file")
ap.add_argument("-s", "--skip-processed", required=False, default=False, help="Skips already processed images")
ap.add_argument("-l", "--log-file", required=False, default='pipeline.log', help="Log file")
ap.add_argument("-w", "--nb-workers", required=False, default='1', help="Number of workers for parallelization")
args = vars(ap.parse_args())

skip_processed = args['skip_processed']

##################################################
# Getting raws folder and checking for existence #
##################################################
raws_path = args['raws']
raws_folder = Path(raws_path)
if not raws_folder.exists():
    raise Exception("Raws files not found under %s" % raws_folder)
raws_folder = raws_folder.resolve()
if raws_folder.is_dir():
    raw_files = glob.glob('{}/*'.format(raws_folder))
else:
    assert raws_folder.is_file()
    raw_files = []
    with raws_folder.open('r') as f:
        raw_files = f.read().split('\n')
        raw_files = [e for e in raw_files if e != '']


#############################################
# Getting destination folder or creating it #
#############################################
destination_path = args['destination']
destination_folder = Path(destination_path)
if not destination_folder.exists():
    destination_folder.mkdir()
destination_folder = destination_folder.resolve()

####################
# Reading Log File #
####################
processed_images = []
processed_file = destination_folder / VISITED_LOG_FILE_NAME
if processed_file.exists():
    with processed_file.open() as f:
        processed_images = [line.rstrip('\n') for line in f]

########################
# Loading the TF model #
########################
model_path = args['model']
m = Model(model_path, 11, 3, 3)
print("Got Model")

##########################
# Looping over raw scans #
##########################


def process_one(file):
    print(file)
    file = Path(file)
    filename = file.name
    recto = RECTO_SUBSTRING_JPG in filename
    if not recto:
        return
    base_path = re.sub(RECTO_SUBSTRING_JPG, '', str(file))
    name = re.sub(RECTO_SUBSTRING_JPG, '', filename)
    # name = re.sub('_', '', base_path)
    if name in processed_images and skip_processed:
        return

    current_folder = destination_folder / name
    if not current_folder.exists():
        os.makedirs(str(current_folder))

    processed_successfully = True

    ####################
    # VERSO PROCESSING #
    ####################
    #doc_info = DocumentInfo(base_path, side='verso')

    #try:
    #    verso_raw_scan = RawScan(doc_info, base_path)
    #    verso_raw_scan.crop_cardboard(m)

    #    verso_raw_scan.save_prediction(str(current_folder / 'prediction.jpg'))

    #    cardboard = verso_raw_scan.get_cardboard()
    #    cardboard.save_image(str(current_folder / 'cardboard.jpg'))

    #except Exception as e:
    #    processed_successfully = False
    #    doc_info.logger.error(e)

    ####################
    # RECTO PROCESSING #
    ####################
    doc_info = DocumentInfo(base_path, side='recto')

    try:

        recto_raw_scan = RawScan(doc_info, base_path)
        recto_raw_scan.crop_cardboard(m)
        recto_raw_scan.crop_image()

        recto_raw_scan.save_prediction(str(current_folder / 'prediction.jpg'))

        cardboard = recto_raw_scan.get_cardboard()
        cardboard.save_image(str(current_folder / 'cardboard.jpg'))

        extracted_image = recto_raw_scan.get_image()
        extracted_image.save_image(str(current_folder / 'image.jpg'))

        doc_info.logger.debug("Done!")
    except Exception as e:
        processed_successfully = False
        doc_info.logger.error(e)


    ######################
    # Saving to log file #
    ######################
    if name not in processed_images and processed_successfully:
        processed_images.append(name)
        with processed_file.open('a') as f:
            f.write(name + '\n')

nb_workers = int(args['nb_workers'])

log_file = args['log_file']
if os.path.exists(log_file):
    raise IOError('Log file "{}" already exists'.format(log_file))

logger = logging.getLogger()
fhandler = logging.FileHandler(filename=log_file, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)

for file in raw_files:
    process_one(file)

