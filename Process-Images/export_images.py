from text_extraction import Rectangle
import os
from PIL import Image
import uuid
import shutil
from glob import glob
import argparse
from tqdm import tqdm
import json

BASE_DHCANVAS_URL = 'http://cini.dhlab.epfl.ch/page/view/{collection_uuid}/p{page_number}'
def dhcanvas_url_from_full_id(_id: str):
    box_id, sequence_id = _id.split('_')
    return BASE_DHCANVAS_URL.format(collection_uuid=box_uuid_from_id(box_id), page_number=int(sequence_id))

def box_uuid_from_id(_id: str):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, 'DHCANVAS_CINI_{}'.format(_id)))

BASE_IIIF_URL = 'http://dhlabsrv4.epfl.ch/iiif_cini'
def make_page_from_folder(folder):
    cardboard_path = os.path.join(folder, 'cardboard.jpg')
    ocr_path = os.path.join(folder, 'ocr_complete.json')
    if not (os.path.exists(cardboard_path) and os.path.exists(ocr_path)):
        return None
    try:
        result_data = {}
        img = Image.open(cardboard_path)
        result_data['width'], result_data['height'] = img.size[0], img.size[1]
        result_data['metadata'] = {}
        box_id, result_data['id'] = folder.split('/')[-2:]
        result_data['url'] = '{}/{}%2F{}.jpg'.format(BASE_IIIF_URL, box_id, result_data['id'])
        result_data['uuid'] = str(uuid.uuid5(uuid.NAMESPACE_URL, result_data['url']))
        fragments = Rectangle.load_from_json(ocr_path)

        result_data['sequenceNumber'] = int(result_data['id'].split('_')[1])

        result_data['segments'] = []
        for fm in fragments:
            result_data['segments'].append({
                'x': fm.x1,
                'y': fm.y1,
                'h': fm.y2-fm.y1,
                'w': fm.x2-fm.x1,
                'transcription': fm.text,
                'label': fm.label,
                'uuid': str(uuid.uuid4())
            })
        return result_data
    except Exception as e:
        return None


def make_document_from_folder(folder):
    image_folders = glob('{}/*'.format(folder))
    image_folders = [f for f in image_folders if os.path.isdir(f)]
    filtered_image_folders = []
    for f in image_folders:
        try:
            int(f.split('_')[-1])
            filtered_image_folders.append(f)
        except ValueError as verr:
            pass
    image_folders = filtered_image_folders
    image_folders = sorted(image_folders, key=lambda f: int(f.split('_')[-1]))
    result_data = dict()
    result_data['id'] = os.path.basename(folder)
    result_data['metadata'] = {}
    result_data['uuid'] = box_uuid_from_id(result_data['id'])
    result_data['pages'] = []
    for i, f in enumerate(image_folders):
        tmp = make_page_from_folder(f)
        if tmp is not None:
            result_data['pages'].append(tmp)
    return result_data


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", required=True, help="Folder with the extracted cardboards")
    ap.add_argument("-c", "--cardboards-dir", required=True, help="Where to save the output cardboards")
    ap.add_argument("-i", "--images-dir", required=True, help="Where to save the output images")
    args = vars(ap.parse_args())

    input_dir = args['directory']
    cardboards_dir = args['cardboards_dir']
    images_dir = args['images_dir']

    os.makedirs(cardboards_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    folders = glob('{}/**/'.format(input_dir), recursive=True)
    folders = [os.path.relpath(f, input_dir) for f in folders if os.path.isdir(f)]

    for folder in tqdm(folders):
        input_file = os.path.join(input_dir, folder, 'cardboard.jpg')
        if os.path.exists(input_file):
            output_file = os.path.join(cardboards_dir, folder + '.jpg')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            shutil.copyfile(input_file, output_file)
        input_file = os.path.join(input_dir, folder, 'image.jpg')
        if os.path.exists(input_file):
            output_file = os.path.join(images_dir, folder + '.jpg')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            shutil.copyfile(input_file, output_file)
