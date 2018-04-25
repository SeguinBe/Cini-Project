# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import argparse
import os
from text_extraction import Rectangle, AUTHOR_LABEL, DESCRIPTION_LABEL
from random import shuffle

import json
import numpy as np
from glob import glob


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input-folder", required=True, help="Folder with the json file to be corrected")
ap.add_argument("-o", "--output-folder", required=True, help="Folder with the saved corrected json files")
args = ap.parse_args()

INPUT_FOLDER = args.input_folder
OUTPUT_FOLDER = args.output_folder
os.makedirs(args.output_folder, exist_ok=True)

#input_elements = {os.path.basename(f): f for f in glob(os.path.join(INPUT_FOLDER, '*.json'))}
input_elements = sorted([os.path.basename(f) for f in glob(os.path.join(INPUT_FOLDER, '*.json'))])
shuffle(input_elements)


def _get_dropdown_label_values():
    result = []
    for basename in input_elements:
        tmp = {
            'label': basename,
            'value': basename
        }
        if os.path.exists(os.path.join(OUTPUT_FOLDER, basename)):
            tmp['label'] += ' (Done)'
        result.append(tmp)
    return result


def _get_latest_transcription(basename):
    saved_file = os.path.join(OUTPUT_FOLDER, basename)
    if os.path.exists(saved_file):
        with open(saved_file, 'r') as f:
            return json.load(f)
    else:
        input_file = os.path.join(INPUT_FOLDER, basename)
        rects = Rectangle.load_from_json(input_file)
        return {
            AUTHOR_LABEL: next((r.text for r in rects if r.label == AUTHOR_LABEL), ''),
            DESCRIPTION_LABEL: next((r.text for r in rects if r.label == DESCRIPTION_LABEL), '')
        }

def _save_transcription(basename, author_txt, description_txt):
    saved_file = os.path.join(OUTPUT_FOLDER, basename)
    with open(saved_file, 'w') as f:
            json.dump({
                AUTHOR_LABEL: author_txt,
                DESCRIPTION_LABEL: description_txt
            }, f)

app = dash.Dash()

txt_area_style = {
    'width': '60%',
    'height': '100px',
    'font-size': '18px'
}

app.layout = html.Div(children=[
    html.H1(children='Transcription Validation System'),
    dcc.Dropdown(
        id='cardboard-dropdown',
        options=_get_dropdown_label_values(),
        value=None
    ),
    html.Img(id='image'),
    html.Div([
        dcc.Textarea(
            id='author-input',
            placeholder='Author Field',
            style=txt_area_style
        ),
    ]),
    html.Div([
        dcc.Textarea(
            id='description-input',
            placeholder='Description Field',
            style=txt_area_style
        ),
    ]),
    html.Div([
        html.Button(id='button', children='Save', style={'font-size': '30px'})
    ])
])


@app.callback(
    Output(component_id='image', component_property='src'),
    [Input(component_id='cardboard-dropdown', component_property='value')]
)
def update_output(input_value):
    input_value = input_value.replace('.json', '')
    drawer_id, cardboard_number = input_value.split('_')
    return 'http://dhlabsrv4.epfl.ch/iiif_cini/{}%2F{}.jpg/pct:0,0,100,25/!1000,1000/0/default.jpg'.format(drawer_id, input_value)


@app.callback(
    Output(component_id='author-input', component_property='value'),
    [Input(component_id='cardboard-dropdown', component_property='value')]
)
def update_output(input_value):
    return _get_latest_transcription(input_value).get(AUTHOR_LABEL, '')


@app.callback(
    Output(component_id='description-input', component_property='value'),
    [Input(component_id='cardboard-dropdown', component_property='value')]
)
def update_output(input_value):
    return _get_latest_transcription(input_value).get(DESCRIPTION_LABEL, '')


@app.callback(
    Output(component_id='cardboard-dropdown', component_property='options'),
    [Input(component_id='button', component_property='n_clicks')],
    state=[
        State(component_id='cardboard-dropdown', component_property='value'),
        State(component_id='description-input', component_property='value'),
        State(component_id='author-input', component_property='value')
    ]
)
def save_transcription(n_clicks, basename, description_txt, author_txt):
    if n_clicks > 0:
        _save_transcription(basename, author_txt, description_txt)

    return _get_dropdown_label_values()


app.run_server(debug=True, host='0.0.0.0')
