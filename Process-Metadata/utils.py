import re
import numpy as np


def to_wd_identifier(e):
    _tmp = 'http://www.wikidata.org/entity/'
    assert 'value' in e.keys()
    assert e['type'] == 'uri'
    assert _tmp == e['value'][:len(_tmp)]
    return 'wd:'+e['value'][len(_tmp):]


def to_ulan_identifier(e):
    _tmp = 'http://vocab.getty.edu/ulan/'
    assert 'value' in e.keys()
    assert e['type'] == 'uri'
    assert _tmp == e['value'][:len(_tmp)]
    return 'ulan:'+e['value'][len(_tmp):]


def to_url(e: str):
    if e.startswith('wd:'):
        return e.replace('wd:', 'http://www.wikidata.org/entity/')
    elif e.startswith('ulan:'):
        return e.replace('ulan:', 'http://vocab.getty.edu/ulan/')
    raise NotImplementedError


def to_string(e):
    assert 'value' in e.keys()
    assert e['type'] == 'literal'
    return e['value']


def read_words_list(filename):
    with open(filename, 'r') as f:
        elements = f.readlines()
        elements = map(lambda s: s.replace('\n', ''), elements)
        elements = filter(lambda s: len(s) > 0, elements)
    return list(elements)


def read_pairs_list(filename):
    elements = read_words_list(filename)
    elements_pairs = [s.split('\t') for s in elements]
    elements_pairs = filter(lambda s: len(s) == 2, elements_pairs)
    elements_pairs = filter(lambda s: len(s[1]) > 0 and len(s[0]) > 0, elements_pairs)
    return list(elements_pairs)


_useless_words = read_words_list('data/useless_words.txt')


def normalized_str(s, advanced=True):
    s = s.lower()
    s = re.sub(r"[,;\-\.\n\(\)']", ' ', s)
    if advanced:
        s = re.sub(r'\b(11|1l|l1|i1|1i|il)\b', '', s)
        for w in _useless_words:
            s = re.sub(r'\b{}\b'.format(w), '', s)
    else:
        s = re.sub(r'\b(11|1l|l1|i1|1i|il)\b', 'il', s)
    s = re.sub(' +', ' ', s)
    return s.strip()


def final_str(s):
    s = re.sub(r"[;\.\n]", ' ', s)
    s = re.sub(r'\b(11|1l|l1|i1|1i|il)\b', 'il', s)
    s = re.sub(' +', ' ', s)
    return s.strip()


def normalized_ngrams(s):
    return frozenset(normalized_str(s).split())