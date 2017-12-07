import argparse
import glob
import os.path
import re
from pathlib import Path
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from doc_seg.loader import LoadedModel
from traceback import print_exc, format_exc
import cv2

from raw_scan import RawScan
from base import *
from shared import *

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--raws", required=True, help="Folder with raw images or text files with filenames to be processed")
ap.add_argument("-d", "--destination", required=True, help="Folder where the results will be saved")
ap.add_argument("-m", "--model", required=True, help="The model file")
ap.add_argument("-s", "--skip-processed", action='store_true', help="Skips already processed images")
ap.add_argument("-l", "--log-file", required=False, default='pipeline.log', help="Log file")
ap.add_argument("-w", "--nb-workers", required=False, default='1', help="Number of workers for parallelization")
ap.add_argument("-g", "--gpu", required=False, default='0', help="Number of workers for parallelization")
ap.add_argument("--subset", required=False, default='', help="Multiple.Subset for instance 3.1 for only processing id % 3 == 1")
args = vars(ap.parse_args())

skip_processed = args['skip_processed']
if args['subset'] != '':
    subset = tuple(int(i) for i in args['subset'].split('.'))
else:
    subset = None

#cv2.setNumThreads(6)

##################################################
# Getting raws folder and checking for existence #
##################################################
raws_path = args['raws']
raws_folder = Path(raws_path)
if not raws_folder.exists():
    raise Exception("Raws files not found under %s" % raws_folder)
raws_folder = raws_folder.resolve()
raw_files = glob.glob('{}/**/*.jpg'.format(raws_folder), recursive=True)
# if raws_folder.is_dir():
#     raw_files = glob.glob('{}/**/*.jpg'.format(raws_folder), recursive=True)
# else:
#     assert raws_folder.is_file()
#     raw_files = []
#     with raws_folder.open('r') as f:
#         raw_files = f.read().split('\n')
#         raw_files = [e for e in raw_files if e != '']


#############################################
# Getting destination folder or creating it #
#############################################
destination_path = args['destination']
destination_folder = Path(destination_path)
if not destination_folder.exists():
    destination_folder.mkdir()
destination_folder = destination_folder.resolve()

########################
# Loading the TF model #
########################
model_path = args['model']
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, visible_device_list=args['gpu'])
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)).as_default():
    m = LoadedModel(model_path)
print("Got Model")

##########################
# Looping over raw scans #
##########################

pbar = tqdm(total=len(raw_files))
def monitor_finish(fn):
    def _fn(*args, **kwargs):
        r = fn(*args, **kwargs)
        pbar.update(1)
        return r
    return _fn

@monitor_finish
def process_one(file):
    try:
        file = Path(file)
        filename = file.name
        recto = RECTO_SUBSTRING_JPG in filename
        if not recto:
            return
        base_path = re.sub(RECTO_SUBSTRING_JPG, '', str(file))
        relative_base_path = os.path.relpath(base_path, str(raws_folder))
        name = re.sub(RECTO_SUBSTRING_JPG, '', filename)
        # name = re.sub('_', '', base_path)

        doc_info = DocumentInfo(base_path, side='recto')

        if subset:
            cardboard_id = int(name.split('_')[1])
            if cardboard_id % subset[0] != subset[1]:
                return

        current_folder = destination_folder / relative_base_path
        if not current_folder.exists():
            os.makedirs(str(current_folder))

        if skip_processed and os.path.exists(str(current_folder / 'cardboard.jpg')) \
                and os.path.exists(str(current_folder / 'image.jpg')):
            return

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

        with CatchTime('RawScan creation'):
            recto_raw_scan = RawScan(doc_info, base_path)
        with CatchTime('Cardboard + Image cropping'):
            recto_raw_scan.crop_cardboard(m)

        with CatchTime('Saving files'):
            recto_raw_scan.save_prediction(str(current_folder / 'prediction.jpg'))
            recto_raw_scan.save_extraction(str(current_folder / 'extraction.jpg'))

            cardboard = recto_raw_scan.get_cardboard()
            cardboard.save_image(str(current_folder / 'cardboard.jpg'))

            extracted_image = recto_raw_scan.get_image()
            extracted_image.save_image(str(current_folder / 'image.jpg'))

        doc_info.logger.debug("Done!")
    except Exception as e:
        print(format_exc())
        doc_info.logger.error(e)


nb_workers = int(args['nb_workers'])

log_file = args['log_file']
#if os.path.exists(log_file):
#    raise IOError('Log file "{}" already exists'.format(log_file))

logger = logging.getLogger()
fhandler = logging.FileHandler(filename=log_file, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)

if nb_workers > 1:
    with ThreadPoolExecutor(nb_workers) as e:
        e.map(process_one, raw_files)
else:
    for f in raw_files:
        process_one(f)
