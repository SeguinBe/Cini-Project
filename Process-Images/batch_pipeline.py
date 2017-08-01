from subprocess import Popen

INPUT_FOLDER = '/scratch/benoit/cini_full_images'
OUTPUT_FOLDER = '/scratch/benoit/cini_processed'
MODEL_FOLDER = '/tmp/tensorboard2/unet_v2/export'
LOG_FILE = '/home/seguin/cini_extraction-subset.log'

NB_PROCESSES = 6

for i in range(NB_PROCESSES):
    Popen(["python", "pipeline.py",
           '-r', INPUT_FOLDER,
           '-d', OUTPUT_FOLDER,
           '-m', MODEL_FOLDER,
           '-l', LOG_FILE+str(i),
           '--subset', '{}.{}'.format(NB_PROCESSES, i)])
