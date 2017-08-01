from collections import OrderedDict
from PIL import Image
import os
import json


def make_manifest(MANIFEST_SERVER_URL, collection_path, manifest_path, base_path, metadata, iiif_fields,
                  image_paths, image_path_to_iiif_url):
    """

    :param MANIFEST_SERVER_URL:
    :param collection_path:
    :param manifest_path:
    :param base_path:
    :param metadata:
    :param iiif_fields: (required) label, description  (optional) attribution, logo, license, related
    :param image_paths:
    :param image_path_to_iiif_url:
    :return:
    """

    manifest_url = "{}/{}".format(MANIFEST_SERVER_URL, manifest_path)
    sequence_url = "{}/{}/sequence/normal".format(MANIFEST_SERVER_URL, base_path)
    thumbnail = {
        "@id": "{}/full/300,/0/default.jpg".format(image_path_to_iiif_url(image_paths[0])),
        "service": {
            "@context": "http://iiif.io/api/image/2/context.json",
            "@id": image_path_to_iiif_url(image_paths[0]),
            "profile": "http://iiif.io/api/image/2/level1.json"
        },
        "@type": "dctypes:Image"
    }

    manifest = OrderedDict()
    manifest["@context"] = "http://iiif.io/api/presentation/2/context.json"
    manifest["@id"] = manifest_url
    manifest["@type"] = "sc:Manifest"

    manifest["metadata"] = [{'label':k, 'value':v} for k, v in metadata.items()]
    manifest["thumbnail"] = thumbnail

    # "viewingDirection": "right-to-left",
    # "viewingHint": "paged",
    # "navDate": "1856-01-01T00:00:00Z",

    # "license": "http://rightsstatements.org/vocab/NoC-NC/1.0/",

    assert 'label' in iiif_fields.keys()
    assert 'description' in iiif_fields.keys()
    for k, v in iiif_fields.items():
        manifest[k] = v

    manifest["within"] = "{}/{}".format(MANIFEST_SERVER_URL, collection_path)

    manifest["sequences"] = [
        {
            "@id": sequence_url,
            "@type": "sc:Sequence",
            "label": "Normal Page Order",

            "canvases": []
        }
    ]
    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        width, height = img.size[0], img.size[1]

        canvas_url = "{}/{}/canvas/image{}".format(MANIFEST_SERVER_URL, base_path, i + 1)
        annotation_url = "{}/{}/annotation/paint-image{}".format(MANIFEST_SERVER_URL, base_path, i + 1)
        resource_url = "{}/full/!800,800/0/default.jpg".format(image_path_to_iiif_url(image_path))
        thumbnail = {
            "@id": "{}/full/300,/0/default.jpg".format(image_path_to_iiif_url(image_path)),
            "service": {
                "@context": "http://iiif.io/api/image/2/context.json",
                "@id": image_path_to_iiif_url(image_path),
                "profile": "http://iiif.io/api/image/2/level1.json"
            },
            "@type": "dctypes:Image"
        }
        manifest["sequences"][0]["canvases"].append({
                "@context": "http://iiif.io/api/presentation/2/context.json",
                "@id": canvas_url,
                "@type": "sc:Canvas",
                "label": "View #{}".format(i+1),
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
                                "@id": image_path_to_iiif_url(image_path),
                                "profile": "http://iiif.io/api/image/2/level2.json"
                            },
                            "height": height,
                            "width": width
                        },
                        "on": canvas_url
                    }
                ]

            })
    return manifest


def save_manifest(MANIFEST_SERVER_URL, manifest, output_folder):
    manifest_url = manifest['@id']
    rel_path = os.path.relpath(manifest_url, MANIFEST_SERVER_URL)
    output_file = os.path.join(output_folder, rel_path)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Save
    with open(output_file, 'w') as f:
        json.dump(manifest, f)


def make_collection_manifest(MANIFEST_SERVER_URL, collection_path, iiif_fields,
                             collection_manifest_urls, element_manifest_urls):

    colection_manifest = OrderedDict()
    colection_manifest["@context"] = "http://iiif.io/api/presentation/2/context.json"
    colection_manifest["@id"] = "{}/{}".format(MANIFEST_SERVER_URL, collection_path)
    colection_manifest["@type"] = "sc:Collection"
    assert 'label' in iiif_fields.keys()
    assert 'description' in iiif_fields.keys()
    for k, v in iiif_fields.items():
        colection_manifest[k] = v

    if collection_manifest_urls and len(collection_manifest_urls) > 0:
        colection_manifest["collections"] = [
            {
                "@id": manifest_url,
                "@type": "sc:Collection"
            } for manifest_url in collection_manifest_urls
        ]
    if element_manifest_urls and len(element_manifest_urls) > 0:
        colection_manifest["manifests"] = [
            {
                "@id": manifest_url,
                "@type": "sc:Manifest"
            } for manifest_url in element_manifest_urls
        ]
    return colection_manifest
