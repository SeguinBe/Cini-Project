import argparse
import os
import json
from text_extraction import Rectangle
from PIL import Image
from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from export_to_dhcanvas import dhcanvas_url_from_full_id, get_sorted_image_folders, get_sorted_collection_folders
from text_extraction import AUTHOR_LABEL, DESCRIPTION_LABEL

MANIFEST_BASE_URL = 'http://data.dhlab.epfl.ch/cini'
BASE_IIIF_URL = 'http://dhlabsrv4.epfl.ch'
CINI_LOGO_URL = 'http://www.cidim.it:8080/dwnld/bwbsc/image/241898/MARCHIO_LOGO.jpg'
CINI_ATTRIBUTION = 'Fondazione Giorgio Cini'


def get_replica_url(folder: str):
    box_id, element_id = folder.split('/')[-2:]
    return '{}/iiif_replica/cini%2F{}%2F{}.jpg'.format(BASE_IIIF_URL, box_id, element_id)


def get_metadata(folder: str):
    ocr_path = os.path.join(folder, 'ocr_complete.json')
    fragments = Rectangle.load_from_json(ocr_path)
    return [
        {"label": r.label if r.label is not None else 'Unknown', "value": r.text} for r in fragments
        ]


def manifest_path(box_id, manifest_id):
    return 'manifest/{}/{}.json'.format(box_id, manifest_id)


def collection_path(box_id):
    return 'collection/{}.json'.format(box_id)


def get_collection_url(box_id):
    return "{}/{}".format(MANIFEST_BASE_URL, collection_path(box_id))


def make_image_manifest(folder: str, sequence_number: int):
    image_path = os.path.join(folder, 'image.jpg')
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

        manifest["related"] = dhcanvas_url_from_full_id(os.path.basename(folder), sequence_number)
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


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", required=True, help="Folder with the extracted cardboards")
    ap.add_argument("-o", "--output-dir", required=True, help="Where to save the output")
    args = vars(ap.parse_args())

    input_dir = args['directory']
    output_dir = args['output_dir']
    os.makedirs(output_dir)

    folders = get_sorted_collection_folders(args['directory'])

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