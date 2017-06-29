import tensorflow as tf
import cv2
import numpy as np
import math
import os

from skimage import measure
from scipy import ndimage


def get_cleaned_cardboard_prediction(prediction):

    # Performe Erosion and Dilation
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.medianBlur(cv2.morphologyEx(prediction.astype(np.uint8), cv2.MORPH_OPEN, kernel), 5)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

    # Flip classes
    closing[closing == 0] = 3
    closing[closing == 1] = 0
    closing[closing == 3] = 1

    # Upsace prediction
    closing = ndimage.interpolation.zoom(input=closing, zoom=2, order=0)

    # Get enclosing rect and rotate
    angle = cv2.minAreaRect(np.argwhere(closing > 0))[2]
    while angle > 45:
        angle -= 90
    while angle < -45:
        angle += 90
    closing = rotate_image(closing, angle)
    return closing, angle


def uwrap(closing):
    rect = cv2.boundingRect(np.argwhere(closing > 0))
    # Get edges
    edges = cv2.Canny(closing.astype(np.uint8), 0, 5) - cv2.Canny(closing.astype(np.uint8), 0, 6)
    edges = measure.label(edges)
    edges = get_largest_component(edges)

    # Split edges in 4 different matrices and find the distortion params
    p, center_x, center_y = find_distortion_params(*separate_sides(edges, rect))

    return p, center_x, center_y


def rotate_image (image, angle):
    """
    Takes an image and a angle and roteates the image by the given angle while maintaining the whole image
    
    :param image: as an np matrix
    :param angle: float angle
    :return: roated image matrix
    """

    height, width = image.shape[:2]
    image_center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(image_center, angle * -1, 1)

    radians = math.radians(angle * -1)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    M[0, 2] += ((bound_w / 2) - image_center[0])
    M[1, 2] += ((bound_h / 2) - image_center[1])

    return cv2.warpAffine(image, M, (bound_w, bound_h))


def get_largest_component(mat):
    max_sum = 0
    max_comp = 0
    i = 1

    while True:
        sum = np.sum(mat == i)
        if sum > max_sum:
            max_sum = sum
            max_comp = i
        if sum == 0:
            break
        i += 1

    mat[mat != max_comp] = 0
    mat[mat == max_comp] = 1

    return mat


def separate_sides(edges, rect):

    edge_list = np.argwhere(edges > 0)

    center_x = (np.max(edge_list[:, 1]) + np.min(edge_list[:, 1])) // 2
    center_y = (np.max(edge_list[:, 0]) + np.min(edge_list[:, 0])) // 2

    top = []
    left_side = []
    right_side = []
    bottom = []

    for i in range(len(edge_list)):

        x = edge_list[i, 1]
        y = edge_list[i, 0]

        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[0] + rect[2]
        y2 = rect[1]
        x3 = center_x
        y3 = center_y

        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        c = 1 - a - b
        if (0 <= a and a <= 1 and 0 <= b and b <= 1 and 0 <= c and c <= 1):
            top.append(edge_list[i])
            continue
        x2 = rect[0]
        y2 = rect[1] + rect[3]
        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        c = 1 - a - b
        if (0 <= a and a <= 1 and 0 <= b and b <= 1 and 0 <= c and c <= 1):
            left_side.append(edge_list[i])
            continue
        x1 = rect[0] + rect[2]
        y1 = rect[1] + rect[3]
        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        c = 1 - a - b
        if (0 <= a and a <= 1 and 0 <= b and b <= 1 and 0 <= c and c <= 1):
            bottom.append(edge_list[i])
            continue
        x2 = rect[0] + rect[2]
        y2 = rect[1]
        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        c = 1 - a - b
        if (0 <= a and a <= 1 and 0 <= b and b <= 1 and 0 <= c and c <= 1):
            right_side.append(edge_list[i])
            continue

    return np.stack(top), np.stack(bottom), np.stack(left_side), np.stack(right_side), rect, center_x, center_y


def find_distortion_params(top, bottom, left_side, right_side, rect, center_x, center_y):

    slice_places = [len(top), len(top) + len(bottom), len(top) + len(bottom) + len(left_side),
                    len(top) + len(bottom) + len(left_side) + len(right_side)]

    input_coor = np.concatenate([top, bottom, left_side, right_side])

    rec = [rect[1], rect[1] + rect[3], rect[0], rect[0] + rect[2]]

    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    with tf.Graph().as_default():

        def poly_func(x, y, k_x, p_x, k_y, p_y, mid_x, mid_y):

            radius = tf.add(tf.square(x - mid_x), tf.square(y - mid_y)) / (
                (tf.reduce_max(x) - tf.reduce_min(x)) * (tf.reduce_max(y) - tf.reduce_min(y)))

            coef_x = 1 + tf.multiply(radius, k_x[0]) + tf.multiply(tf.square(radius), k_x[1])
            coef_y = 1 + tf.multiply(radius, k_y[0]) + tf.multiply(tf.square(radius), k_y[1])

            add_x = ((p_x[1] * (radius + (2 * tf.square(x)))) + (2 * p_x[0] * x * y)) * (
                1 + tf.multiply(radius, p_x[2]) + tf.multiply(tf.square(radius), p_x[3]))
            add_y = ((p_y[1] * (radius + (2 * tf.square(y)))) + (2 * p_y[0] * x * y)) * (
                1 + tf.multiply(radius, p_y[2]) + tf.multiply(tf.square(radius), p_y[3]))

            return tf.multiply(x, coef_x) + add_x, tf.multiply(y, coef_y) + add_y

        def inference(coor_x, coor_y, mid_x, mid_y):
            return poly_func(coor_x, coor_y, k_x, p_x, k_y, p_y, mid_x, mid_y)

        def loss(coor_x, coor_y, true_x, true_y, rec, slc):
            y_diff = tf.reduce_sum(tf.abs(tf.slice(coor_y, [0, 0], [1, slc[0]]) - rec[0])) / tf.cast(slc[0],
                                                                                                     tf.float32)
            y_diff1 = tf.reduce_sum(tf.abs(tf.slice(coor_y, [0, slc[0]], [1, slc[1] - slc[0]]) - rec[1])) / tf.cast(
                slc[1] - slc[0], tf.float32)
            y_diff2 = tf.reduce_sum(tf.abs(coor_y[slc[1]:slc[2]] - true_y[slc[1]:slc[2]])) / tf.cast(
                slc[2] - slc[1], tf.float32)
            y_diff3 = tf.reduce_sum(tf.abs(coor_y[slc[2]:slc[3]] - true_y[slc[2]:slc[3]])) / tf.cast(
                slc[3] - slc[2], tf.float32)
            x_diff = tf.reduce_sum(tf.abs(coor_x[0:slc[0]] - true_x[0:slc[0]])) / tf.cast(slc[0], tf.float32)
            x_diff1 = tf.reduce_sum(tf.abs(coor_x[slc[0]:slc[1]] - true_x[slc[0]:slc[1]])) / tf.cast(
                slc[1] - slc[0], tf.float32)
            x_diff2 = tf.reduce_sum(tf.abs(tf.slice(coor_x, [0, slc[1]], [1, slc[2] - slc[1]]) - rec[2])) / tf.cast(
                slc[2] - slc[1], tf.float32)
            x_diff3 = tf.reduce_sum(tf.abs(tf.slice(coor_x, [0, slc[2]], [1, slc[3] - slc[2]]) - rec[3])) / tf.cast(
                slc[3] - slc[2], tf.float32)
            return tf.add_n([y_diff, y_diff1, x_diff2, x_diff3])

        def training(loss):
            with tf.name_scope("train"):
                train_step = tf.train.GradientDescentOptimizer(tf.constant(1e-14)).minimize(loss)
            return train_step

        def get_params():
            return k_x, p_x, k_y, p_y

        with tf.device('/cpu:0'):
            k_x = tf.Variable(tf.random_uniform([2], minval=0, maxval=0))
            p_x = tf.Variable(tf.random_uniform([4], minval=0, maxval=0))

            k_y = tf.Variable(tf.random_uniform([2], minval=0, maxval=0))
            p_y = tf.Variable(tf.random_uniform([4], minval=0, maxval=0))

            pl_x = tf.placeholder(tf.float32, [1, len(input_coor)])
            pl_y = tf.placeholder(tf.float32, [1, len(input_coor)])
            rec_coor = tf.placeholder(tf.float32, [4])
            slice_place = tf.placeholder(tf.int32, [4])

            mid_x = tf.placeholder(tf.float32)
            mid_y = tf.placeholder(tf.float32)

            out = inference(pl_x, pl_y, mid_x, mid_y)
            loss_var = loss(out[0], out[1], pl_x, pl_y, rec_coor, slice_place)
            train_op = training(loss_var)
            get_p = get_params()

        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = ""
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(30):
                sess.run(train_op, feed_dict={pl_x: (input_coor[:, 1]).reshape(1, -1),
                                              pl_y: (input_coor[:, 0]).reshape(1, -1), mid_x: center_x,
                                              mid_y: center_y, rec_coor: rec, slice_place: slice_places})

            p = sess.run(get_p, feed_dict={pl_x: (input_coor[:, 1]).reshape(1, -1),
                                           pl_y: (input_coor[:, 0]).reshape(1, -1), mid_x: center_x,
                                           mid_y: center_y, rec_coor: rec, slice_place: slice_places})

        return p, center_x, center_y


