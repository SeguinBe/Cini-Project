import numpy as np
from utils import read_words_list

def _get_secs():
    _secs = [
        ('II', 100, 200),
        ('III', 200, 300),
        ('IV', 300, 400),
        ('V', 400, 500),
        ('VI', 500, 600),
        ('VII', 600, 700),
        ('VIII', 700, 800),
        ('IX', 800, 900),
        ('X', 900, 1000),
        ('XI', 1000, 1100),
        ('XII', 1100, 1200),
        ('XIII', 1200, 1300),
        ('XIV', 1300, 1400),
        ('XV', 1400, 1500),
        ('XVI', 1500, 1600),
        ('XVII', 1600, 1700),
        ('XVIII', 1700, 1800),
        ('XIX', 1800, 1900),
        ('XX', 1900, 2000)
    ]

    l = len(_secs)
    for i in range(l):
        if i < l-1:
            _secs.append(('{}-{}'.format(_secs[i][0], _secs[i+1][0]), _secs[i][1]+50, _secs[i+1][2]-50))
        _secs.append(("{} in".format(_secs[i][0]), _secs[i][1], _secs[i][1]+50))
        _secs.append(("{} ex".format(_secs[i][0]), _secs[i][1]+50, _secs[i][1]+100))
        _secs.append(("{} m".format(_secs[i][0]), _secs[i][1]+25, _secs[i][1]+75))
    return _secs


def get_unknown_entities():

    elements = read_words_list('data/unknown.txt')
    element_pairs = [(s.lower().replace(' ', '-'), s.strip()) for s in elements]

    additional_entities = dict()
    # No date information
    additional_entities.update({
        'cini:{}'.format(p): {
            'label': p,
            'alternateLabels': [],
            'nationality': 'Unknown',
            'gender': 'Unknown'
        } for p_key, p in element_pairs
    })

    # Date information present (i.e PITTORE SEC XI)
    additional_entities.update({
        'cini:{}_{}'.format(p_key, s.replace(' ', '_')): {
            'label': p+(' ' if len(p) > 0 else '')+'SEC '+s,
            'alternateLabels': [],
            'beginRangeDate': s_b,
            'endRangeDate': s_e,
            'nationality': 'Unknown',
            'gender': 'Unknown'
        } for p_key, p in element_pairs+[('unknown', '')] for s, s_b, s_e in _get_secs()
    })

    return additional_entities
