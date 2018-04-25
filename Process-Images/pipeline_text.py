import argparse
import glob
import os.path
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool
from base import DocumentInfo
import numpy as np
from PIL import Image
from text_extraction import Rectangle, cut_text_section_and_resize, detect_text, words_to_fragments,\
    label_fragments, rescale_fragments
from itertools import islice

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="Folder with the extracted cardboards")
ap.add_argument("-w", "--nb-workers", required=False, default='1', help="Number of workers for parallelization")
ap.add_argument("-s", "--skip-processed", action='store_true', help="Skips already processed images")
ap.add_argument("-l", "--log-file", required=False, default='pipeline.log', help="Log file")
ap.add_argument("--force-google-ocr", action='store_true', help="Force reperforming OCR")
args = vars(ap.parse_args())

skip_processed = args['skip_processed']
force_ocr = args['force_google_ocr']

dir_path = args['directory']
#cardboard_files = list(tqdm(glob.iglob('{}/**/cardboard.jpg'.format(dir_path), recursive=True)))
cardboard_files = glob.iglob('{}/**/cardboard.jpg'.format(dir_path), recursive=True)


pbar = tqdm()#total=len(cardboard_files))
def monitor_finish(fn):
    def _fn(*args, **kwargs):
        r = fn(*args, **kwargs)
        pbar.update(1)
        return r
    return _fn


#@monitor_finish
def process_one(file):
    rel_dir = os.path.dirname(file)
    base_path = os.path.split(rel_dir)[-1]
    doc_info = DocumentInfo(base_path, side='recto')
    try:
        ocr_raw_file = os.path.join(rel_dir, 'ocr_raw.json')
        complete_ocr_file = os.path.join(rel_dir, 'ocr_complete.json')

        assert os.path.exists(file)
        img = Image.open(file)
        if force_ocr or not os.path.exists(ocr_raw_file):
            img2 = cut_text_section_and_resize(np.asarray(img))
            words = detect_text(img2)
            Rectangle.save_to_json(words, ocr_raw_file)
            doc_info.logger.debug("Saved raw ocr result")

        if (not skip_processed) or not os.path.exists(complete_ocr_file):
            words = Rectangle.load_from_json(ocr_raw_file)
            words = words[1:]  # Discard the first result as being the one with all the text
            fragments = words_to_fragments(words)
            labelled_fragments = label_fragments(fragments)
            rescaled_fragments = rescale_fragments(labelled_fragments, img.size[0])
            Rectangle.save_to_json(rescaled_fragments, complete_ocr_file)
            doc_info.logger.debug("Saved complete ocr result")

    except Exception as e:
        doc_info.logger.error(e)


nb_workers = int(args['nb_workers'])

log_file = args['log_file']

logger = logging.getLogger()
fhandler = logging.FileHandler(filename=log_file, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)


def split_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))


if nb_workers > 1:
    with Pool(nb_workers) as p:
        for simple_result in tqdm(p.imap(process_one, cardboard_files, chunksize=50)):
            pass
    #for chunk in split_every(2000, cardboard_files):
    #    with ProcessPoolExecutor(nb_workers) as e:
    #        e.map(process_one, chunk, chunksize=20)
else:
    for f in tqdm(cardboard_files):
        process_one(f)
