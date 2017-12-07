from SPARQLWrapper import SPARQLWrapper, JSON, N3, TURTLE, RDF
from collections import defaultdict
from time import sleep
from tqdm import tqdm
import pickle
import pandas as pd
from utils import to_string, to_ulan_identifier, to_url, to_wd_identifier


def get_alternate_names_ulan(entities_id):
    sparql = SPARQLWrapper("http://vocab.getty.edu/sparql")
    sparql.setQuery("""
    SELECT DISTINCT ?p ?alias
    WHERE
    {
      VALUES ?p {
      """ + " ".join(entities_id) +
      """
      }.
      ?p (xl:altLabel|xl:prefLabel)/gvp:term ?alias .
      #VALUES ?l {xl:altLabel xl:prefLabel}
    }
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return [(to_ulan_identifier(r['p']), to_string(r['alias'])) for r in results['results']['bindings']]


def get_ulan_entities():
    import requests
    import io
    # Getting the csv directly (the query is the same) because we do not get all of them otherwise with json rdf
    getty_csv_query = "http://vocab.getty.edu/sparql.csv?query=SELECT+%3Fp+%3FpLabel+%3Fbirthdate+%3Fdeathdate+%3Fnationality+%3Fgender+WHERE+%7B%0D%0A++++ulan%3A500000002+skos%3Amember+%3Fp+.%0D%0A++++%3Fp+gvp%3AprefLabelGVP%2Fgvp%3Aterm+%3FpLabel+.%0D%0A++++++%3Fp+foaf%3Afocus+%3Fagent+.%0D%0A++++%3Fagent+gvp%3AbiographyPreferred%2Fgvp%3AestStart+%3Fbirthdate+.%0D%0A++++%3Fagent+gvp%3AbiographyPreferred%2Fgvp%3AestEnd+%3Fdeathdate+.%0D%0A++++%3Fagent+gvp%3AbiographyPreferred%2Fschema%3Agender%2Fxl%3AprefLabel%2Fgvp%3Aterm+%3Fgender+.%0D%0A++++%3Fagent+gvp%3AnationalityPreferred%2Fxl%3AprefLabel%2Fgvp%3Aterm+%3Fnationality+.%0D%0A++%09FILTER+%28lang%28%3Fnationality%29+%3D+%27en%27%29+.%0D%0A++%09FILTER+%28lang%28%3Fgender%29+%3D+%27en%27%29%0D%0A++++%23FILTER%28+%3Fbirthdate+%3C+1900%29%0D%0A++%7D&_implicit=false&implicit=true&_equivalent=false&_form=%2Fsparql"
    s = requests.get(getty_csv_query).content
    getty_raw_data = pd.read_csv(io.StringIO(s.decode('utf-8')))
    #sparql = SPARQLWrapper("http://vocab.getty.edu/sparql")
    #sparql.setQuery("""
        #SELECT ?p ?pLabel ?birthdate ?deathdate ?nationality ?gender WHERE {
        #ulan:500000002 skos:member ?p .
        #?p gvp:prefLabelGVP/gvp:term ?pLabel .
        #  ?p foaf:focus ?agent .
        #?agent gvp:biographyPreferred/gvp:estStart ?birthdate .
        #?agent gvp:biographyPreferred/gvp:estEnd ?deathdate .
        #?agent gvp:biographyPreferred/schema:gender/xl:prefLabel/gvp:term ?gender .
        #?agent gvp:nationalityPreferred/xl:prefLabel/gvp:term ?nationality .
        #FILTER (lang(?nationality) = 'en') .
        #FILTER (lang(?gender) = 'en')
        ##FILTER( ?birthdate < 1900)
    #}""")
    #sparql.setReturnFormat(JSON)
    #results = sparql.query().convert()
    #entities = {to_ulan_identifier(r['p']) : {'label': to_string(r['pLabel']), 'birthDate': int(to_string(r['birthdate']))}
    #            for r in results['results']['bindings'] }
    entities = {r.p.replace('http://vocab.getty.edu/ulan/', 'ulan:'):
                {'label': r.pLabel, 'birthDate': int(r.birthdate), 'deathDate': int(r.deathdate), 'nationality': r.nationality, 'gender': r.gender}
                for _, r in getty_raw_data.iterrows()}
    entities = {k: v for k, v in entities.items() if v['birthDate'] < 1900}

    batch_size = 400
    all_alternate_namings = defaultdict(list)
    all_ids = list(entities.keys())
    for i in tqdm(range(0, len(entities)//batch_size + 1)):
        id_aliases_pairs = get_alternate_names_ulan(all_ids[i*batch_size:(i+1)*batch_size])
        if len(id_aliases_pairs) == 0:
            break
        for _id, alternate_alias in id_aliases_pairs:
            all_alternate_namings[_id].append(alternate_alias)

    for _id in all_ids:
        entities[_id]['alternateLabels'] = list(set(all_alternate_namings[_id]))
    return entities


def get_alternate_names_wd(entities_id):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery("""
    SELECT DISTINCT ?p ?alias
    WHERE
    {
      VALUES ?p {
      """
            + " ".join(entities_id) +
    """
      }.
      ?p skos:altLabel ?alias FILTER (LANG (?alias) = "en")
    }
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return [(to_wd_identifier(r['p']), to_string(r['alias'])) for r in results['results']['bindings']]


def get_wd_entities():
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery("""
    SELECT ?p ?pLabel WHERE {
      {
        SELECT ?p WHERE {
      ?p wdt:P106 ?class .
      FILTER (?class IN (wd:Q1028181, wd:Q1281618, wd:Q329439, wd:Q42973)) .  # Painter, engraver, sculptor, architect
      #BIND (wd:Q5592 AS ?p) .
      ?p p:P569/psv:P569 ?birth_date_node .
       ?birth_date_node wikibase:timeValue ?birth_date .
       FILTER (year(?birth_date) < 1900)
      } GROUP BY ?p
      } .
      #?p skos:altLabel ?alias FILTER (LANG (?alias) = "en")
      SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en,it,fr,de"
      }
    }""")
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    entities = {to_wd_identifier(r['p']): {'label': to_string(r['pLabel'])}
                for r in results['results']['bindings']}

    batch_size = 400
    all_alternate_namings = defaultdict(list)
    all_ids = list(entities.keys())
    for i in tqdm(range(0, len(entities)//batch_size + 1)):
        id_aliases_pairs = get_alternate_names_wd(all_ids[i*batch_size:(i+1)*batch_size])
        if len(id_aliases_pairs) == 0:
            break
        for _id, alternate_alias in id_aliases_pairs:
            all_alternate_namings[_id].append(alternate_alias)

    for _id in all_ids:
        entities[_id]['alternateLabels'] = list(set(all_alternate_namings[_id]))
    return entities


if __name__ == '__main__':
    ulan_entities = get_ulan_entities()
    wd_entities = get_wd_entities()
    with open('wd_ulan_data.pkl', 'wb') as f:
        pickle.dump((wd_entities, ulan_entities), f)