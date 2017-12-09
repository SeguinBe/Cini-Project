import argparse
from glob import glob
import json
from tqdm import tqdm
import pandas as pd


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", required=True, help="Directory with the processed data.")
    args = ap.parse_args()
    input_directory = args.directory

    files = glob('{}/**/**/ocr_complete.json'.format(input_directory))

    all_data = []
    for filename in tqdm(files):
        # Load the json
        with open(filename) as f:
            data = json.load(f)
        result = {d['label']: d['text'] for d in data if 'label' in d.keys()}
        all_data.append(result)

    df = pd.DataFrame.from_records(all_data)
    df.to_csv('gathered_metadata.csv')