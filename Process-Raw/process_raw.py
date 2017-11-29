import argparse
import os

from multiprocessing import Pool
import md5
import numpy as np
from PIL import Image
from rawkit.raw import Raw
from glob import iglob
import time
import logging

from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--raws", required=True, help="Folder with raw image files")
ap.add_argument("-d", "--destination", required=True, help="Folder where the results will be saved")
ap.add_argument("-w", "--nbworkers", required=False, default='1', help="Number of workers for parallelization")
ap.add_argument("-s", "--skip-processed", action='store_true', help="Skips already processed images")
ap.add_argument("--md5", action='store_true', help="Flag for checking md5 or not")
ap.add_argument("--verso", action='store_true', help="Flag for converting verso files as well")
args = vars(ap.parse_args())

skip_processed = args['skip_processed']

##################################################
# Getting raws folder and checking for existence #
##################################################

input_directory = args['raws']
if not os.path.exists(input_directory):
    print("Folder does not exist")
    exit(1)

output_directory = args['destination']
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

logger = logging.getLogger('ProcessRaw')
hdlr = logging.FileHandler(os.path.join(output_directory, "out-{}.log".format(str(time.time()))))
formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)


class File:
    def __init__(self, path, save_path, check_md5=True):
        self.verify_md5 = check_md5
        self.path = path
        self.save_path = save_path


def process_file(file: File):
    """
    This function takes a file name of a raw image file with an extenstion .cr2 and saves
    a converted image in the defined save path povided to this script. The funcion will also verify the md5 hash
    if required

    Args:
        :param (File): Path to the file

    Returns:
        :return: File name of processed image or an error message

    Note:
        Function specifically coded to work with images form the Replica project. Error messages can be traced in the 
        log file

    """

    try:
        if os.path.exists(file.save_path) and skip_processed:
            return file
        if file.verify_md5:
            if md5.check_md5(file.path, file.path.replace(".cr2", ".md5")):
                logger.debug("{0} valid md5".format(file.path))
            else:
                logger.error("{0} invalid md5".format(file.path))

        with Raw(file.path) as raw_image:
            buffered_image = np.array(raw_image.to_buffer())

            os.makedirs(os.path.dirname(file.save_path), exist_ok=True)

            buffered_image = buffered_image.reshape((raw_image.metadata.height, raw_image.metadata.width, 3))
            if 'verso' in file.path:
                buffered_image = buffered_image.reshape((raw_image.metadata.width, raw_image.metadata.height, 3))
                buffered_image = np.rot90(buffered_image, k=-1)
            else:
                buffered_image = buffered_image.reshape((raw_image.metadata.height, raw_image.metadata.width, 3))
            image = Image.fromarray(buffered_image)
            #image = Image.frombytes('RGB', (raw_image.metadata.height, raw_image.metadata.width), buffered_image)

            image.save(file.save_path, format='jpeg', quality=90)
            logger.info("Done processing  {0}".format(file.path))

    except Exception as e:
        logger.exception("{} excepted with error".format(file.path))


######################################
# Setting params and load file names #
######################################

max_workers = int(args["nbworkers"])
verify_md5 = args['md5']
inputs = []
all_results = []

if args['verso']:
    matching_filenames = os.path.join(input_directory, '**', '*.cr2')
else:
    matching_filenames = os.path.join(input_directory, '**', '*recto.cr2')

for file in tqdm(iglob(matching_filenames, recursive=True), desc="Indexing files"):
    relative_path = file[len(input_directory):]
    inputs.append(File(os.path.join(input_directory, relative_path), os.path.join(output_directory, relative_path).replace('.cr2', '.jpg'), verify_md5))


######################
# Start main process #
######################
with Pool(max_workers) as p:
    for simple_result in tqdm(p.imap(process_file, inputs, chunksize=5), total=len(inputs)):
        all_results.append(simple_result)
