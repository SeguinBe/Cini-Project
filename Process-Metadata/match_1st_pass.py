import pickle
from utils import normalized_ngrams, normalized_str, read_words_list, read_pairs_list, final_str
from cini_specifics import get_unknown_entities
from collections import defaultdict
import pandas as pd
import argparse
import numpy as np


def match(input_names, id_name_pairs, ambiguous_clearing_data):

    normalized_name_dict = defaultdict(set)
    normalized_ngrams_dict = defaultdict(set)
    for _id, name in id_name_pairs:
        #print(name)
        normalized_name_dict[normalized_str(name)].add(_id)
        normalized_ngrams_dict[normalized_ngrams(name)].add(_id)

    matched = dict()
    ambiguous = dict()
    failed = list()

    ambiguous_clearing_data = {normalized_str(k): v for k, v in ambiguous_clearing_data.items()}

    for s in input_names:
        try:
            if pd.isnull(s):
                continue
            n = normalized_str(s)
            if normalized_str(s) in normalized_name_dict.keys():
                values = normalized_name_dict[normalized_str(s)]
            elif normalized_ngrams(s) in normalized_ngrams_dict.keys():
                values = normalized_ngrams_dict[normalized_ngrams(s)]
            else:
                values = []

            if len(values) == 1:
                match = list(values)[0]
                #print('SUCCESS, matched {} with {}:{}'.format(s, match, entities.get(match, {'label': match})['label']))
                matched[s] = match
            elif len(values) == 0:
                #print('FAILED, could not match {}'.format(s))
                failed.append(s)
            else:
                #print('AMBIGUOUS, matched {} with {}'.format(s, values))
                if n in ambiguous_clearing_data.keys():
                    matched[s] = ambiguous_clearing_data[n]
                else:
                    ambiguous[s] = values
        except Exception as e:
            print('ERROR, {}'.format(s))
    return matched, ambiguous, failed


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True, help="Csv file with the openrefined data")
    args = ap.parse_args()
    input_filename = args.data

    with open('wd_ulan_data.pkl', 'rb') as f:
        wd_entities, ulan_entities = pickle.load(f)
    entities = ulan_entities
    print("Loaded {} entities".format(len(entities)))

    # Read additional alternate labels
    additional_namings = read_pairs_list('data/alternate_labels.txt')
    for ulan_id, name in additional_namings:
        entities['ulan:{}'.format(ulan_id)]['alternateLabels'].append(name)
    print("Loaded and added {} additional names".format(len(additional_namings)))
    # Read disambiguation data for specific cases
    ambiguous_clearing_data = read_pairs_list('data/disambiguation.txt')
    ambiguous_clearing_data = {name: 'ulan:{}'.format(ulan_id) for name, ulan_id in ambiguous_clearing_data}
    print("Loaded {} disambiguation cases".format(len(ambiguous_clearing_data)))
    # Generate unknown situations
    unknown_entities = get_unknown_entities()
    print("Loaded and generated {} unknwon cases".format(len(unknown_entities)))
    # Gather everything
    all_entities = {**entities, **unknown_entities}
    id_name_pairs = [(_id, n) for _id, e in all_entities.items() for n in set(e['alternateLabels']+[e['label']])]
    additional_pairs = []
    for _id, n in id_name_pairs:
        for to_be_replaced, replacement in read_pairs_list('data/generative_substitutions.txt'):
            if to_be_replaced in n:
                additional_pairs.append((_id, n.replace(to_be_replaced, replacement)))
    id_name_pairs.extend(additional_pairs)
    print("Generated {} possible id-label pairs".format(len(id_name_pairs)))

    # Read names
    df = pd.read_csv(input_filename)
    input_names = df.Author.values
    matched, ambiguous, failed = match(input_names, id_name_pairs, ambiguous_clearing_data)
    nb_total = len(input_names)
    nb_almost_empty = np.sum([a == np.nan or len(str(a)) < 5 or len(str(a)) > 400 for a in input_names])
    print("Total {}, almost empty {}|{:.1f}%".format(nb_total, nb_almost_empty, 100*nb_almost_empty/nb_total))

    # Compute candidates for 2nd pass
    candidates = defaultdict(list)
    for a in input_names:
        if a in matched.keys():
            candidates[normalized_str(a, advanced=False)].append(final_str(a))
    candidates = {k: pd.Series(v).value_counts().index[0]  # Take the most common
                  for k, v in candidates.items()}

    for k, v in unknown_entities.items():  # Add the generated so they are used as candidates
        matched[v['label']] = k
        candidates[normalized_str(v['label'], advanced=False)] = v['label']

    matched_final = dict()
    for a in matched.keys():
        matched_final[a] = {'name': candidates[normalized_str(a, advanced=False)],
                            'id': matched[a]}

    nb_matched = len([a for a in input_names if a in matched.keys()])
    print("Matched: {}|{:.1f}%|{:.1f}%".format(nb_matched, 100*nb_matched/nb_total,
                                               100*nb_matched/(nb_total-nb_almost_empty)))
    nb_ambiguous = len([a for a in input_names if a in ambiguous.keys()])
    print("Ambigous match: {}|{:.1f}%|{:.1f}%".format(nb_ambiguous, 100*nb_ambiguous/nb_total,
                                                      100*nb_ambiguous/(nb_total-nb_almost_empty)))

    with open('match_1st_pass.pkl', 'wb') as f:
        pickle.dump((matched_final, ambiguous, failed), f)
