import argparse
import os
from glob import glob
import json
import jellyfish
from text_extraction import Rectangle, AUTHOR_LABEL, DESCRIPTION_LABEL
import re
import pandas as pd
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input-folder", required=True, help="Folder with the json file")
ap.add_argument("-o", "--output-folder", required=True, help="Folder with the saved corrected json files")
args = ap.parse_args()

INPUT_FOLDER = args.input_folder
OUTPUT_FOLDER = args.output_folder

input_elements = sorted([os.path.basename(f) for f in glob(os.path.join(OUTPUT_FOLDER, '*.json'))])


def get_transcription(basename, groundtruth: bool):
    saved_file = os.path.join(OUTPUT_FOLDER, basename)
    if groundtruth:
        with open(saved_file, 'r') as f:
            return json.load(f)
    else:
        input_file = os.path.join(INPUT_FOLDER, basename)
        rects = Rectangle.load_from_json(input_file)
        return {
            AUTHOR_LABEL: next((r.text for r in rects if r.label == AUTHOR_LABEL), ''),
            DESCRIPTION_LABEL: next((r.text for r in rects if r.label == DESCRIPTION_LABEL), '')
        }


def normalized_str(s):
    s = s.lower()
    s = re.sub(r"[,;\-\.\n\(\)']", ' ', s)
    s = re.sub(' +', ' ', s)
    return s.strip()


results = []
for basename in tqdm(input_elements):
    gt_transcription = get_transcription(basename, groundtruth=True)
    input_transcription = get_transcription(basename, groundtruth=False)
    gt_author, gt_description = gt_transcription[AUTHOR_LABEL], gt_transcription[DESCRIPTION_LABEL]
    extracted_author, extracted_description = input_transcription[AUTHOR_LABEL], input_transcription[DESCRIPTION_LABEL]
    # print(gt_author, gt_description, extracted_author, extracted_description)
    try:
        results.append({
            'basename': basename,
            'author_error': jellyfish.damerau_levenshtein_distance(gt_author, extracted_author),
            'description_error': jellyfish.damerau_levenshtein_distance(gt_description, extracted_description),
            'author_len': len(gt_author),
            'description_len': len(gt_description),
            'author_error_normalized': jellyfish.damerau_levenshtein_distance(normalized_str(gt_author),
                                                                              normalized_str(extracted_author)),
            'description_error_normalized': jellyfish.damerau_levenshtein_distance(normalized_str(gt_description),
                                                                                   normalized_str(
                                                                                       extracted_description))
        })
        if jellyfish.damerau_levenshtein_distance(normalized_str(gt_author), normalized_str(extracted_author))>0:
            print(gt_author, extracted_author)
    except Exception:
        print(basename)

df = pd.DataFrame.from_records(results)

print('CER (author) : {:.2f}'.format(100 * df.author_error.sum() / df.author_len.sum()))
print('CER (description) : {:.2f}'.format(100 * df.description_error.sum() / df.description_len.sum()))
print('CER (author, normalized) : {:.2f}'.format(100 * df.author_error_normalized.sum() / df.author_len.sum()))
print('CER (description, normalized) : {:.2f}'.format(
    100 * df.description_error_normalized.sum() / df.description_len.sum()))

print('Perfect transcription (author) : {:.2f}'.format(100 * (df.author_error == 0).sum() / len(df)))
print('Perfect transcription (description) : {:.2f}'.format(100 * (df.description_error == 0).sum() / len(df)))
print('Perfect transcription (author, normalized) : {:.2f}'.format(
    100 * (df.author_error_normalized == 0).sum() / len(df)))
print('Perfect transcription (description, normalized) : {:.2f}'.format(
    100 * (df.description_error_normalized == 0).sum() / len(df)))

print('1-away transcription (author) : {:.2f}'.format(100 * (df.author_error <= 1).sum() / len(df)))
print('1-away  transcription (description) : {:.2f}'.format(100 * (df.description_error <= 1).sum() / len(df)))
print('1-away  transcription (author, normalized) : {:.2f}'.format(
    100 * (df.author_error_normalized <= 1).sum() / len(df)))
print('1-away  transcription (description, normalized) : {:.2f}'.format(
    100 * (df.description_error_normalized <= 1).sum() / len(df)))
