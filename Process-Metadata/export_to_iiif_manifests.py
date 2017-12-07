import argparse
import os
import json
#from text_extraction import Rectangle
from PIL import Image
from glob import glob
from tqdm import tqdm
from collections import OrderedDict
import pickle
import numpy as np
#from export_to_dhcanvas import dhcanvas_url_from_full_id, get_sorted_image_folders, get_sorted_collection_folders

AUTHOR_LABEL = 'Author'
DESCRIPTION_LABEL = 'Description'
MATCHING_DIR = dict()

MANIFEST_BASE_URL = 'http://data.dhlab.epfl.ch/cini'
MANIFEST_BASE_URL_CARDBOARD = 'http://data.dhlab.epfl.ch/cini_raw'
CARDBOARD_MANIFEST = False
BASE_IIIF_URL = 'http://dhlabsrv4.epfl.ch'
CINI_LOGO_URL = 'http://www.cidim.it:8080/dwnld/bwbsc/image/241898/MARCHIO_LOGO.jpg'
CINI_ATTRIBUTION = 'Fondazione Giorgio Cini'


class Rectangle:
    def __init__(self, y1, y2, x1, x2, text=None, label=None):
        self.label = label
        self.text = text
        self.y1 = y1
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2
        self.arr = np.array([y1, y2, x1, x2])

    def __repr__(self):
        return "Rect(y:{},{}|x:{},{}{}{})".format(self.y1, self.y2, self.x1, self.x2,
                                                "|{}".format(repr(self.text)) if self.text is not None else "",
                                                "|{}".format(repr(self.label)) if self.label is not None else "")
    @classmethod
    def from_dict(cls, input_dict):
        return Rectangle(**input_dict)

    @classmethod
    def load_from_json(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return [Rectangle.from_dict(d) for d in data]


def get_sorted_image_folders(box_folder: str):
    def _id_ordering(_id: str):
        if _id.isdigit():
            return int(_id)
        if _id[:-1].isdigit():
            return int(_id[:-1]) + (ord(_id[-1])+1)/256
        raise ValueError('Could not parse image number {} in box {}'.format(_id, box_folder))
    folders = glob('{}/*'.format(box_folder))
    # Filter non directories and elements which do not match the proper ordering
    folders = [f for f in folders if os.path.isdir(f) and _id_ordering(f.split('_')[-1]) is not None]
    return sorted(folders, key=lambda n: _id_ordering(n.split('_')[-1]))


def get_sorted_collection_folders(box_folder: str):
    def _id_ordering(_id: str):
        if _id.isdigit():
            return int(_id)
        if _id[:-1].isdigit():
            return int(_id[:-1]) + (ord(_id[-1])+1)/256
    folders = glob('{}/*'.format(box_folder))
    # Filter non directories and elements which do not match the proper ordering
    folders = [f for f in folders if os.path.isdir(f) and _id_ordering(f.split('/')[-1]) is not None]
    return sorted(folders, key=lambda n: _id_ordering(n.split('/')[-1]))


def get_replica_url(folder: str):
    box_id, element_id = folder.split('/')[-2:]
    if CARDBOARD_MANIFEST:
        return '{}/iiif_cini/{}%2F{}.jpg'.format(BASE_IIIF_URL, box_id, element_id)
    else:
        return '{}/iiif_replica/cini%2F{}%2F{}.jpg'.format(BASE_IIIF_URL, box_id, element_id)


def get_metadata(folder: str):
    ocr_path = os.path.join(folder, 'ocr_complete.json')
    fragments = Rectangle.load_from_json(ocr_path)
    metadata = {
         r.label if r.label is not None else 'Unknown': r.text for r in fragments
    }
    if AUTHOR_LABEL in metadata.keys():
        author = metadata[AUTHOR_LABEL]
        if author in MATCHING_DIR.keys():
            d = MATCHING_DIR[author]
            metadata['AuthorAligned'] = d['author_corrected_name']
            if 'id' in d.keys():
                metadata['AuthorId'] = d['id']
            if 'author_url' in d.keys():
                metadata['AuthorURL'] = d['author_url']
            if 'begin_date' in d.keys():
                metadata['DateRangeBegin'] = d['begin_date']
            if 'end_date' in d.keys():
                metadata['DateRangeEnd'] = d['end_date']
    return [
        {"label": k, "value": v} for k, v in metadata.items()
        ]


def manifest_path(box_id, manifest_id):
    return 'manifest/{}/{}.json'.format(box_id, manifest_id)


def collection_path(box_id):
    return 'collection/{}.json'.format(box_id)


def get_collection_url(box_id):
    return "{}/{}".format(MANIFEST_BASE_URL, collection_path(box_id))


def make_image_manifest(folder: str, sequence_number: int):
    if not CARDBOARD_MANIFEST:
        image_path = os.path.join(folder, 'image.jpg')
    else:
        image_path = os.path.join(folder, 'cardboard.jpg')
    ocr_path = os.path.join(folder, 'ocr_complete.json')
    if not (os.path.exists(image_path) and os.path.exists(ocr_path)):
        return None
    # TODO rajouter la license et l'attribution
    manifest_id = folder.split('/')[-1]
    box_id, number_of_element = manifest_id.split('_')

    manifest_url = "{}/{}".format(MANIFEST_BASE_URL, manifest_path(box_id, manifest_id))
    collection_url = get_collection_url(box_id)
    sequence_url = "{}/{}/sequence/normal".format(MANIFEST_BASE_URL, manifest_id)
    canvas_url = "{}/{}/canvas/image".format(MANIFEST_BASE_URL, manifest_id)
    annotation_url = "{}/{}/annotation/paint-image".format(MANIFEST_BASE_URL, manifest_id)
    resource_url = "{}/{}/res/image.jpg".format(MANIFEST_BASE_URL, manifest_id)
    thumbnail = {
        "@id": "{}/full/300,/0/default.jpg".format(get_replica_url(folder)),
        "service": {
            "@context": "http://iiif.io/api/image/2/context.json",
            "@id": get_replica_url(folder),
            "profile": "http://iiif.io/api/image/2/level1.json"
        },
        "@type": "dctypes:Image"
    }
    try:
        img = Image.open(image_path)
        width, height = img.size[0], img.size[1]
        metadata = get_metadata(folder)
        # Extract the author and description fields from the metadata (if any)
        author, description = 'Unknown Artist', 'Untitled'  # Default values
        for d in metadata:
            if d['label'] == DESCRIPTION_LABEL:
                description = d['value']
            elif d['label'] == AUTHOR_LABEL:
                author = d['value']

        manifest = OrderedDict()
        manifest["@context"] = "http://iiif.io/api/presentation/2/context.json"
        manifest["@id"] = manifest_url
        manifest["@type"] = "sc:Manifest"
        manifest["label"] = "Image {} from box {} from the Cini fototeca".format(number_of_element, box_id)
        manifest["metadata"] = metadata
        manifest["description"] = "{},\nby {}".format(description, author)
        manifest["thumbnail"] = thumbnail

        # "viewingDirection": "right-to-left",
        # "viewingHint": "paged",
        # "navDate": "1856-01-01T00:00:00Z",

        # "license": "http://rightsstatements.org/vocab/NoC-NC/1.0/",
        manifest["attribution"] = CINI_ATTRIBUTION
        manifest["logo"] = CINI_LOGO_URL

        #manifest["related"] = dhcanvas_url_from_full_id(os.path.basename(folder), sequence_number)
        if not CARDBOARD_MANIFEST:
            manifest["related"] = 'http://universalviewer.io/uv.html?manifest={}/{}'.format(MANIFEST_BASE_URL_CARDBOARD, manifest_path(box_id, manifest_id))
        manifest["within"] = collection_url

        manifest["sequences"] = [
            {
                "@id": sequence_url,
                "@type": "sc:Sequence",
                "label": "Normal Page Order",

                "canvases": [
                    {
                        "@context": "http://iiif.io/api/presentation/2/context.json",
                        "@id": canvas_url,
                        "@type": "sc:Canvas",
                        "label": "Scanned Image",
                        "height": height,
                        "width": width,
                        "thumbnail": thumbnail,
                        "images": [
                            {
                                "@context": "http://iiif.io/api/presentation/2/context.json",
                                "@id": annotation_url,
                                "@type": "oa:Annotation",
                                "motivation": "sc:painting",
                                "resource": {
                                    "@id": resource_url,
                                    "@type": "dctypes:Image",
                                    "format": "image/jpeg",
                                    "service": {
                                        "@context": "http://iiif.io/api/image/2/context.json",
                                        "@id": get_replica_url(folder),
                                        "profile": "http://iiif.io/api/image/2/level2.json"
                                    },
                                    "height": height,
                                    "width": width
                                },
                                "on": canvas_url
                            }
                        ]

                    }

                ]
            }
        ]
        return manifest
    except Exception as e:
        print("Problem with {} : {}".format(folder, e))
        return None


def save_manifest(manifest, output_folder):
    manifest_url = manifest['@id']
    rel_path = os.path.relpath(manifest_url, MANIFEST_BASE_URL)
    output_file = os.path.join(output_folder, rel_path)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Save
    with open(output_file, 'w') as f:
        json.dump(manifest, f)


def process_folder(input_folder: str, output_folder: str):

    # Process the extracted file and save the corresponding manifests
    manifest_list = []
    for i, folder_name in enumerate(get_sorted_image_folders(input_folder)):
        manifest = make_image_manifest(folder_name, i+1)
        if manifest is not None:
            manifest_list.append(manifest['@id'])
            save_manifest(manifest, output_folder)
    # Create the collection manifest
    box_id = input_folder.split('/')[-1]
    collection_manifest = OrderedDict()

    collection_manifest["@context"] = "http://iiif.io/api/presentation/2/context.json"
    collection_manifest["@id"] = get_collection_url(box_id)
    collection_manifest["@type"] = "sc:Collection"
    collection_manifest["label"] = "Drawer {}".format(box_id)

    collection_manifest["description"] = "Images coming from drawer {} of the fototeca of the CINI foundation".format(box_id)
    collection_manifest["attribution"] = CINI_ATTRIBUTION

    collection_manifest["manifests"] = [
        {
            "@id": manifest_url,
            "@type": "sc:Manifest"
        } for manifest_url in manifest_list
    ]

    save_manifest(collection_manifest, output_folder)
    return collection_manifest['@id']


def export(input_dir, output_dir):
    folders = get_sorted_collection_folders(input_dir)

    collection_manifest_urls = []
    for folder in tqdm(folders):
        collection_manifest_urls.append(process_folder(folder, output_dir))

    top_collection_manifest = OrderedDict()
    top_collection_manifest["@context"] = "http://iiif.io/api/presentation/2/context.json"
    top_collection_manifest["@id"] = get_collection_url('top')
    top_collection_manifest["@type"] = "sc:Collection"
    top_collection_manifest["label"] = "Top collection of the CINI foundation"
    top_collection_manifest["description"] = "Top collection indexing the digitized images of the CINI fototeca."
    top_collection_manifest["attribution"] = CINI_ATTRIBUTION

    top_collection_manifest["collections"] = [
        {
            "@id": manifest_url,
            "@type": "sc:Collection"
        } for manifest_url in collection_manifest_urls
        ]

    save_manifest(top_collection_manifest, output_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image-directory", required=True, help="Folder with the extracted images")
    ap.add_argument("-o", "--output-dir", required=True, help="Where to save the output")
    args = vars(ap.parse_args())

    with open('match_final.pkl', 'rb') as f:
        MATCHING_DIR = pickle.load(f)

    input_dir_images = args['image_directory']
    output_dir = args['output_dir']

    output_dir_images = os.path.join(output_dir, 'cini')
    os.makedirs(output_dir_images, exist_ok=True)
    export(input_dir_images, output_dir_images)

    CARDBOARD_MANIFEST = True
    MANIFEST_BASE_URL = MANIFEST_BASE_URL_CARDBOARD
    output_dir_cardboards = os.path.join(output_dir, 'cini_raw')
    os.makedirs(output_dir_cardboards, exist_ok=True)
    export(input_dir_images, output_dir_cardboards)
