import argparse
import pickle
import pandas as pd
from cini_specifics import get_unknown_entities
from tqdm import tqdm
from utils import to_url


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True, help="Csv file with the openrefined data")
    args = ap.parse_args()
    input_filename = args.data

    with open('match_1st_pass.pkl', 'rb') as f:
        matched, ambiguous, failed = pickle.load(f)
    with open('match_2nd_pass.pkl', 'rb') as f:
        matched2 = pickle.load(f)
    with open('wd_ulan_data.pkl', 'rb') as f:
        _, ulan_entities = pickle.load(f)

    entities = {
        **get_unknown_entities(),
        **{k: {'beginRangeDate': v['birthDate']+18, 'endRangeDate': v['deathDate']}
           for k, v in ulan_entities.items()}
    }

    df = pd.read_csv(input_filename)
    df = df[~df.AuthorOriginal.isnull()]

    matching_dict = dict()
    i = 0
    for author_original, author, author_uncertain in tqdm(zip(df.AuthorOriginal.values,
                                                         df.Author.values,
                                                         df.AuthorUncertain), total=len(df)):
        uncertain_assignment = False
        if author in matched.keys():
            author_ulan = matched[author]['id']
            author_corrected_name = matched[author]['name']
            transfer_metadata = True
        elif author in matched2.keys():
            author_ulan = matched2[author]['id']  # type: str
            author_corrected_name = matched2[author]['name']
            transfer_metadata = matched2[author]['transfer_metadata']
            if author_ulan.startswith('cini:'):
                uncertain_assignment = True
        else:
            author_ulan = None
            author_corrected_name = author
            transfer_metadata = False

        if author_uncertain:
            author_corrected_name += ' (?)'
        if uncertain_assignment:
            author_corrected_name += ' *'
        d = {
            'author_corrected_name': author_corrected_name
        }
        if author_ulan is not None:
            d['id'] = author_ulan
            if author_ulan.startswith('ulan:'):
                d['author_url'] = author_ulan.replace('ulan:', 'http://vocab.getty.edu/page/ulan/')
            if 'beginRangeDate' in entities[author_ulan].keys():
                d['begin_date'] = entities[author_ulan]['beginRangeDate']
            if transfer_metadata:
                if 'endRangeDate' in entities[author_ulan].keys():
                    d['end_date'] = entities[author_ulan]['endRangeDate']
                    if d['end_date'] <= d['begin_date']:
                        print(author_ulan, d['begin_date'], d['end_date'])
                        del d['begin_date']
                        del d['end_date']
            i += 1
        matching_dict[author_original] = d

    print('Matched {} out of {} | {:.1f}%'.format(i, len(df), i/len(df)*100))

    with open('match_final.pkl', 'wb') as f:
        pickle.dump(matching_dict, f)
