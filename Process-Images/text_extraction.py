import io
import numpy as np
import cv2
from google.cloud import vision
from functools import total_ordering
import matplotlib.pyplot as plt
from PIL import Image
import json
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from typing import List

_MAX_WIDTH = 2000
_POSSIBLE_HEADER_HEIGHTS = [310, 520, 590]

AUTHOR_LABEL = 'Author'
LOCATION_LABEL = 'City'
INSTITUTION_LABEL = 'Institution'
DESCRIPTION_LABEL = 'Description'
COUNTRY_LABEL = 'Country'
CINI_1_LABEL = 'CiniCollection'
CINI_2_LABEL = 'CiniNumber'
CINI_3_LABEL = 'CiniTime'
REFERENCE_LABEL = 'Reference'
FONDO_STAMP_LABEL = 'FondoStamp'
ALL_LABELS = [AUTHOR_LABEL, LOCATION_LABEL, INSTITUTION_LABEL, DESCRIPTION_LABEL, COUNTRY_LABEL,
CINI_1_LABEL, CINI_2_LABEL, CINI_3_LABEL, REFERENCE_LABEL, FONDO_STAMP_LABEL]


def cut_text_section_and_resize(img, relative_height=0.28):
    new_height = int(img.shape[0] * relative_height)
    output = img[:new_height, :, :]
    if output.shape[1] > _MAX_WIDTH:
        output = cv2.resize(output, (_MAX_WIDTH, int(output.shape[0] * _MAX_WIDTH / output.shape[1])))
    return output


def rescale_fragments(rects, target_width):
    ratio = target_width/_MAX_WIDTH
    rescaled_fragments = [r.scale(ratio, round_values=True) for r in rects]
    return rescaled_fragments


@total_ordering
class Rectangle:
    def __init__(self, y1, y2, x1, x2, text=None, label=None):
        self.label = label
        self.text = text
        self.y1 = y1
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2
        self.arr = np.array([y1, y2, x1, x2])

    @classmethod
    def from_google_annotation(cls, annotation):
        coords = [(v.y, v.x) for v in annotation.bounding_poly.vertices]
        x_coords, y_coords = [v[1] for v in coords], [v[0] for v in coords]
        return Rectangle(min(y_coords), max(y_coords), min(x_coords), max(x_coords), annotation.description)

    @classmethod
    def merge_rectangles(cls, rectangles, line_threshold=20):
        to_be_read = list(rectangles)
        current_rectangle = None
        full_text = ''
        while len(to_be_read) > 0:
            next_line = False
            if current_rectangle is None:
                current_rectangle = min(to_be_read)
            else:
                current_rectangle = min(filter(current_rectangle.is_next, to_be_read),
                                        key=current_rectangle.dist_to_rect,
                                        default=None)
                if current_rectangle is None:
                    current_rectangle = min(to_be_read)
                    next_line = True
            to_be_read.remove(current_rectangle)
            if next_line:
                full_text += '\n'
            else:
                full_text += ' '
            full_text += current_rectangle.text
        full_text = full_text.strip()

        #full_text = ''
        #last_y = None
        #for r in ordained_rectangles:
        #    new_y = r.y2
        #    if last_y is not None:
        #        if new_y - last_y >= line_threshold:
        #            full_text += '\n'
        #        else:
        #            full_text += ' '
        #    last_y = new_y
        #    full_text += r.text

        return Rectangle(min([r.y1 for r in rectangles]), max([r.y2 for r in rectangles]),
                         min([r.x1 for r in rectangles]), max([r.x2 for r in rectangles]),
                         full_text)

    def __lt__(self, other):
        c, o_c = self.center, other.center
        return c[0] * 20 + c[1] < o_c[0] * 20 + o_c[1]

    def __mul__(self, other):
        if isinstance(other, tuple):
            return Rectangle(self.y1*other[0], self.y2*other[0], self.x1*other[1], self.x2*other[1],
                             self.text, self.label)
        else:
            return Rectangle(self.y1*other, self.y2*other, self.x1*other, self.x2*other, self.text, self.label)

    def scale(self, s, round_values=False):
        if isinstance(s, tuple):
            r = Rectangle(self.y1*s[0], self.y2*s[0], self.x1*s[1], self.x2*s[1],
                             self.text, self.label)
        else:
            r = Rectangle(self.y1*s, self.y2*s, self.x1*s, self.x2*s, self.text, self.label)
        if round_values:
            r.y1, r.y2, r.x1, r.x2 = round(r.y1), round(r.y2), round(r.x1), round(r.x2)
        return r

    def __repr__(self):
        return "Rect(y:{},{}|x:{},{}{}{})".format(self.y1, self.y2, self.x1, self.x2,
                                                "|{}".format(repr(self.text)) if self.text is not None else "",
                                                "|{}".format(repr(self.label)) if self.label is not None else "")

    @property
    def center(self):
        return (self.y2 + self.y1) / 2, (self.x2 + self.x1) / 2

    def dist_to_rect(self, rect, skewing=1.0):
        # WARNING this assumes rectangles are not intersecting with each other
        min_y_diff = max(self.y1-rect.y2, rect.y1-self.y2, 0)
        min_x_diff = max(self.x1-rect.x2, rect.x1-self.x2, 0)
        #min_y_diff = np.min(np.abs(self.arr[None, :2] - rect.arr[:2, None]))
        #min_x_diff = np.min(np.abs(self.arr[None, 2:] - rect.arr[2:, None]))
        return np.sqrt(min_y_diff ** 2 + min_x_diff ** 2 / skewing)

    def is_next(self, rect):
        # WARNING this assumes rectangles are not intersecting with each other

        vertical_intersection = max(self.y1, rect.y1) < min(self.y2, rect.y2)
        to_the_rigth = self.x2 < rect.x1
        return vertical_intersection and to_the_rigth

        # end_point = ((self.y2+self.y1)/2, self.x2)
        # next_point = ((rect.y2+rect.y1)/2, rect.x1)
        #alpha = np.arctan2(next_point[0]-end_point[0], next_point[1]-end_point[1])
        #if alpha < -  np.pi:
        #    alpha += 2*np.pi
        #print(alpha)
        #if -0.3 < alpha < 0.3:
        #    return True
        #else:
        #    return False

    def weighted_dist_to_pt(self, pt):
        c_y, c_x = self.center
        dist_y = (c_y-pt[0])/(0.8*(self.y2-self.y1))
        dist_x = (c_x-pt[1])/(0.8*(self.x2-self.x1))
        return np.sqrt(dist_y**2+dist_x**2)

    @classmethod
    def from_dict(cls, input_dict):
        return Rectangle(**input_dict)

    def to_dict(self):
        result = {'y1': self.y1, 'y2': self.y2,
                  'x1': self.x1, 'x2': self.x2}
        if self.text is not None:
            result['text'] = self.text
        if self.label is not None:
            result['label'] = self.label
        return result

    def copy(self):
        return self.from_dict(self.to_dict())

    @classmethod
    def save_to_json(cls, rectangles, filename):
        if isinstance(rectangles, list):
            data = [r.to_dict() for r in rectangles]
        else:
            data = rectangles.to_dict()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return [Rectangle.from_dict(d) for d in data]

    _label_color = {k: 255*v[:3] for k,v in zip(ALL_LABELS, plt.cm.jet(np.arange(len(ALL_LABELS))/(len(ALL_LABELS)-1)))}
    _label_color[None] = np.ones(3)*128
    def draw(self, canvas, print_text=None):
        cv2.rectangle(canvas, (self.x1, self.y1), (self.x2, self.y2), self._label_color.get(self.label, (255, 0, 0)), thickness=3)
        if print_text == 'text':
            cv2.putText(canvas, self.text.replace('\n', '//'), (self.x1, self.y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
        elif print_text == 'label':
            cv2.putText(canvas, self.label, (self.x1, self.y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, self._label_color[self.label])


def _np_array_to_jpg_encoded(img):
    buffer = io.BytesIO()
    Image.fromarray(img).save(buffer, format="JPEG")
    return buffer.getvalue()


def error_assignments(rects: List[Rectangle], gt_rects: List[Rectangle]) -> int:
    errors = 0
    for r in rects:
        # Find match
        m = min(gt_rects, key=lambda g_r: g_r.dist_to_rect(r))
        if r.label != m.label:
            errors += 1
    return errors


def detect_text(img):
    # Instantiates a client
    client = vision.ImageAnnotatorClient()
    g_img = vision.types.Image(content=_np_array_to_jpg_encoded(img))
    r = vision.types.AnnotateImageRequest(image=g_img,
                                          features=[vision.types.Feature(type=vision.types.Feature.TEXT_DETECTION)],
                                          image_context=vision.types.ImageContext(language_hints=['it', 'fr', 'en', 'de']))
    result = client.annotate_image(r)
    # vision_client = vision.Client()
    # g_img = vision_client.image(content=_np_array_to_jpg_encoded(img))
    # result = g_img.detect_text(limit=200)
    fm_list = []
    for anno in result.text_annotations:  # Discard the first result because it is the concatenated version
        fm_list.append(Rectangle.from_google_annotation(anno))
    return fm_list


def words_to_fragments(word_list):
    X = np.zeros((len(word_list), len(word_list)))
    for i, r1 in enumerate(word_list):
        for j, r2 in enumerate(word_list):
            X[i, j] = r1.dist_to_rect(r2)
    cluster_inds = DBSCAN(eps=50, metric="precomputed", min_samples=1).fit_predict(X)
    n_clusters = np.max(cluster_inds) + 1
    fragments = []
    for ind_cluster in range(n_clusters):
        fragments.append(Rectangle.merge_rectangles([word_list[i] for i in np.where(cluster_inds == ind_cluster)[0]]))
    return fragments


FIRST_HORIZONTAL_LINE, SECOND_HORIZONTAL_LINE = 0.3, 0.8

base_rectangles = [
    Rectangle(0, FIRST_HORIZONTAL_LINE, 0, 0.4, label=LOCATION_LABEL),
    Rectangle(0, FIRST_HORIZONTAL_LINE, 0.5, 0.85, label=AUTHOR_LABEL),
    Rectangle(FIRST_HORIZONTAL_LINE, SECOND_HORIZONTAL_LINE, 0, 0.4, label=INSTITUTION_LABEL),
    Rectangle(FIRST_HORIZONTAL_LINE, SECOND_HORIZONTAL_LINE, 0.5, 0.9, label=DESCRIPTION_LABEL)
]
third_line_rectangle = [
    Rectangle(SECOND_HORIZONTAL_LINE, 1, 0, 0.25, label=CINI_1_LABEL),
    Rectangle(SECOND_HORIZONTAL_LINE, 1, 0.25, 0.375, label=CINI_2_LABEL),
    Rectangle(SECOND_HORIZONTAL_LINE, 1, 0.375, 0.5, label=CINI_3_LABEL),
    Rectangle(SECOND_HORIZONTAL_LINE, 1, 0.5, 0.8, label=REFERENCE_LABEL),
]
optional_rectangles = [
    Rectangle(0, FIRST_HORIZONTAL_LINE, 0.4, 0.45, label=COUNTRY_LABEL),
    Rectangle(0, FIRST_HORIZONTAL_LINE, 0.85, 1.0, label=FONDO_STAMP_LABEL)
]


def assign_labels(rectangles, reference_rectangles):
    if len(rectangles) > len(reference_rectangles):
        return None, np.inf
    X = np.zeros((len(rectangles), len(reference_rectangles)))
    for i, r in enumerate(rectangles):
        for j, t_r in enumerate(reference_rectangles):
            X[i, j] = t_r.weighted_dist_to_pt(r.center)

    assignment = linear_sum_assignment(X)
    output = [r.copy() for r in rectangles]
    score = 0
    for i, j in zip(*assignment):
        output[i].label = reference_rectangles[j].label
        score += X[i, j]
    return output, score


def has_author_and_description(labelled_rects):
    has_author, has_description = False, False
    for r in labelled_rects:
        if r.label == AUTHOR_LABEL:
            has_author = True
        if r.label == DESCRIPTION_LABEL:
            has_description = True
    return has_author and has_description


def label_fragments(fragments):
    candidate_layouts = [
        (base_rectangles + third_line_rectangle + optional_rectangles, 1.05),
        (base_rectangles + third_line_rectangle, 1.05),
        #(base_rectangles, SECOND_HORIZONTAL_LINE+0.5),
        #(base_rectangles + optional_rectangles, SECOND_HORIZONTAL_LINE+0.5)
    ]
    target_rectangles = base_rectangles + third_line_rectangle + optional_rectangles
    max_y = max([r.y2 for r in fragments])
    results = []
    for h in _POSSIBLE_HEADER_HEIGHTS:
        results.append(assign_labels(fragments, [r*(h, _MAX_WIDTH) for r in target_rectangles]))

    assigned_rectangles, score = min(results, key=lambda s: s[1])

    assert assigned_rectangles is not None
    return assigned_rectangles


def label_fragments_old(fragments):
    max_y = max([r.y2 for r in fragments])
    target_rectangles = base_rectangles + third_line_rectangle + optional_rectangles
    target_rectangles = [r*(max_y*1.05, _MAX_WIDTH) for r in target_rectangles]
    assigned_rectangles, _ = assign_labels(fragments, target_rectangles)
    if not has_author_and_description(assigned_rectangles) and False:
        target_rectangles = base_rectangles + third_line_rectangle
        target_rectangles = [r*(max_y*1.05, _MAX_WIDTH) for r in target_rectangles]
        assigned_rectangles, _ = assign_labels(fragments, target_rectangles)
        if not has_author_and_description(assigned_rectangles) and False:
            # Assume only 2 lines of elements
            target_rectangles = base_rectangles
            target_rectangles = [r*(max_y*(SECOND_HORIZONTAL_LINE+0.5), _MAX_WIDTH) for r in target_rectangles]
            assigned_rectangles, _ = assign_labels(fragments, target_rectangles)

    assert assigned_rectangles is not None
    return assigned_rectangles