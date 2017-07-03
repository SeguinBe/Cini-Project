import argparse
import os

from multiprocessing import Pool
import md5
import numpy as np
from PIL import Image
from rawkit.raw import Raw
from glob import iglob
import time

from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--raws", required=True, help="Folder with raw image files")
ap.add_argument("-d", "--destination", required=True, help="Folder where the results will be saved")
ap.add_argument("-w", "--nbworkers", required=False, default='1', help="Number of workers for parallelization")
ap.add_argument("--md5", action='store_true', help="Flag for checking md5 or not")
args = vars(ap.parse_args())


##################################################
# Getting raws folder and checking for existence #
##################################################

load_directory = args['raws']
if not os.path.exists(load_directory):
    print("Folder does not exist")
    exit(1)

save_directory = args['destination']
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

f = open(os.path.join(save_directory, "out-{}.log".format(str(time.time()))), 'w')

###############
# Funtion def #
###############
class File:
    def __init__(self, path, save_path, check_md5=True,):
        self.verify_md5 = check_md5
        self.path = path
        self.save_path = save_path

def process_file(file):
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
        if os.path.exists(file.save_path):
            return file
        if file.verify_md5:
            if not md5.check_md5(file.path, file.path.replace(".cr2", ".md5")):
                f.write("{0} invalid md5\n".format(file.path))
                f.flush()
                return "{0} invalid md5".format(file.path)

        with Raw(file.path) as raw_image:
            buffered_image = np.array(raw_image.to_buffer())

            os.makedirs(os.path.dirname(file.save_path), exist_ok=True)
        
            image = Image.frombytes('RGB', (raw_image.metadata.width, raw_image.metadata.height), buffered_image)
            image.save(file.save_path, format='jpeg')
            f.write("{0}\n".format(file.path))
            f.flush()
            return file
        

    except Exception as e:
        f.write("{0} excepted with error: {1}\n".format(file.path, str(e)))
        f.flush()
        return "{0} excepted with error: {1}".format(file.path, str(e))


#####################################
# Setting params and load file names#
#####################################

max_workers = int(args["nbworkers"])
verify_md5 = args['md5']
inputs = []
all_results = []

for file in tqdm(iglob(os.path.join(load_directory, '**/*recto.cr2'), recursive=True), desc="Indexing files"):
    relative_path = file[len(load_directory):]
    inputs.append(File(os.path.join(load_directory, relative_path),  os.path.join(save_directory, relative_path).replace('.cr2', '.jpg'), verify_md5))


######################
# Start main process #
######################

with Pool(max_workers) as p:
    for simple_result in tqdm(p.imap(process_file, inputs, chunksize=5), total=len(inputs)):
        all_results.append(simple_result)


#########################
# Save the logs to file #
#########################
f.flush()
f.close()

