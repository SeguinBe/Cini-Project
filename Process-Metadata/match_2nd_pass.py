from tqdm import tqdm
import jellyfish
import numpy as np
from utils import normalized_str, read_pairs_list
try:
    import dask
    import dask.bag as db
    from dask.diagnostics import ProgressBar
    from dask.distributed import Client, LocalCluster, progress
except Exception as e:
    print('Dask not installed')
import pickle
from multiprocessing import Pool


def get_closest_matches(s, candidates, top_n=1):
    scores = np.array([1-jellyfish.damerau_levenshtein_distance(s, c)/len(s) for c, _ in candidates])
    return [(scores[i], candidates[i]) for i in np.argsort(scores)[-top_n:][::-1]]


def make_candidates(matched, modifiers):
    candidates = dict()
    for n, d in matched.items():
        candidates.update({
            normalized_str(n+(" ({})".format(m) if m else ''), advanced=False): (d, m)
            for m in modifiers
            })
    return list(candidates.items())


def _fn(s, candidates):
    t = normalized_str(s, advanced=False)
    try:
        matches = get_closest_matches(t, candidates)
    except Exception as e:
        matches = [(0, candidates[0])]
    return t, matches


if __name__ == '__main__':
    with open('match_1st_pass.pkl', 'rb') as f:
        matched, ambiguous, failed = pickle.load(f)

    modifiers_dict = {k: bool(int(v)) for k, v in read_pairs_list('data/modifiers.txt')}
    modifiers_dict[None] = True
    candidates = make_candidates(matched, modifiers_dict.keys())
    print('Number of candidates : {}'.format(len(candidates)))

    if True:  # Dask processing
        cluster = LocalCluster(n_workers=48)
        client = Client(cluster)
        b = db.from_sequence(failed, partition_size=200)
        [c] = client.scatter([candidates], broadcast=True)  # Broadcast the list of candidates to the workers
        r = b.map(_fn, c)
        f = client.compute(r)
        progress(f)
        matching_results = f.result()
    else:  # Multiprocessing
        matching_results = []
        with Pool(40) as p:
            for simple_result in tqdm(p.imap(_fn, failed, chunksize=300), total=len(failed)):
                matching_results.append(simple_result)

    matching_results = sorted(matching_results, key=lambda x: x[1][0][0], reverse=True)

    # normalized_str -> (normalized_name, ({'id':_,'name':_}, modifier_str) )
    matching_dict = {r[0]: r[1][0][1] for r in matching_results if r[1][0][0] > 0.88}
    matched2 = dict()
    for s in failed:
        n = normalized_str(s, advanced=False)
        if n in matching_dict.keys():
            d, modifier = matching_dict[n][1]
            matched2[s] = {'name': d['name'] + (" ({})".format(modifier.lower()) if modifier else ''),
                           'id': d['id'],
                           'modifier': modifier.lower() if modifier is not None else None,
                           'transfer_metadata': modifiers_dict[modifier]}

    nb_matched = len([m for m in failed if m in matched2.keys()])
    print("To be matched: {} | Success: {} , {:.1f}%".format(len(failed), nb_matched, 100*nb_matched/len(failed)))

    with open('match_2nd_pass.pkl', 'wb') as f:
        pickle.dump(matched2, f)
