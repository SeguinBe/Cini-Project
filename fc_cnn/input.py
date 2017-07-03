from glob import glob
import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.image import rotate as tf_rotate


def input_fn(input_folder, label_images_folder=None, data_augmentation=False, resized_size=(688, 1024),
             batch_size=1, num_epochs=None, num_threads=4, image_summaries=False):

    # Finding the list of images to be used
    input_images = glob(os.path.join(input_folder, '**', '*.jpg'), recursive=True) + \
                   glob(os.path.join(input_folder, '**', '*.png'), recursive=True)
    print('Found {} images'.format(len(input_images)))

    # Finding the list of labelled images if available
    if label_images_folder:
        label_images = []
        for input_image_filename in input_images:
            label_image_filename = os.path.join(label_images_folder, os.path.basename(input_image_filename))
            if not os.path.exists(label_image_filename):
                raise FileNotFoundError(label_image_filename)
            label_images.append(label_image_filename)

        classes_file = os.path.join(label_images_folder, 'classes.txt')
        if not os.path.exists(classes_file):
            raise FileNotFoundError(classes_file)
        classes_color_values = np.loadtxt(classes_file).astype(np.float32)

    # Helper loading function
    def load_image(filename):
        with tf.name_scope('load_resize_img'):
            decoded_image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
            return decoded_image

    # Tensorflow input_fn
    def fn():
        if not label_images_folder:
            image_filename = tf.train.string_input_producer(input_images, num_epochs=num_epochs).dequeue()
            to_batch = {'images': tf.image.resize_images(load_image(image_filename), resized_size)}
        else:
            # Get one filename of each
            image_filename, label_filename = tf.train.slice_input_producer([input_images, label_images],
                                                                           num_epochs=num_epochs,
                                                                           shuffle=True)
            # Read and resize the images
            label_image = load_image(label_filename)
            input_image = load_image(image_filename)
            # Parallel data augmentation
            if data_augmentation:
                with tf.name_scope('random_flip_lr'):
                    sample = tf.random_uniform([], 0, 1)
                    label_image = tf.cond(sample > 0.5, lambda: tf.image.flip_left_right(label_image), lambda: label_image)
                    input_image = tf.cond(sample > 0.5, lambda: tf.image.flip_left_right(input_image), lambda: input_image)
                with tf.name_scope('random_flip_ud'):
                    sample = tf.random_uniform([], 0, 1)
                    label_image = tf.cond(sample > 0.5, lambda: tf.image.flip_up_down(label_image), lambda: label_image)
                    input_image = tf.cond(sample > 0.5, lambda: tf.image.flip_up_down(input_image), lambda: input_image)
                with tf.name_scope('random_rotate'):
                    rotation_angle = tf.random_uniform([], -0.05, 0.05)
                    label_image = rotate_crop(label_image, rotation_angle, interpolation='NEAREST')
                    input_image = rotate_crop(input_image, rotation_angle, interpolation='BILINEAR')
                input_image = tf.image.random_contrast(input_image, lower=0.8, upper=1.0)
                input_image = tf.image.random_hue(input_image, max_delta=0.1)
                input_image = tf.image.random_saturation(input_image, lower=0.8, upper=1.2)

            input_image = tf.image.resize_images(input_image, resized_size)
            label_image = tf.image.resize_images(label_image, resized_size,
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            to_batch = {'images': input_image, 'labels': label_image_to_class(label_image, classes_color_values)}

        # Batch the preprocessed images
        prepared_batch = tf.train.batch(to_batch, batch_size=batch_size, num_threads=num_threads,
                                        capacity=3*batch_size*num_threads, allow_smaller_final_batch=True)

        # Summaries for checking that the loading and data augmentation goes fine
        if image_summaries:
            tf.summary.image('input/image',
                             tf.image.resize_images(prepared_batch['images'], np.array(resized_size)/3), max_outputs=1)
            if 'labels' in prepared_batch:
                label_export = class_to_label_image(prepared_batch['labels'], classes_color_values)
                tf.summary.image('input/label',
                                 tf.image.resize_images(label_export, np.array(resized_size)/3), max_outputs=1)

        return prepared_batch, prepared_batch.get('labels')

    return fn


def label_image_to_class(label_image, classes_color_values):
    # Convert label_image [H,W,3] to the classes [H,W],int32 according to the classes [C,3]
    with tf.name_scope('LabelAssign'):
        diff = tf.cast(label_image[:, :, None, :], tf.float32) - tf.constant(classes_color_values[None, None, :, :])  # [H,W,C,3]
        pixel_class_diff = tf.reduce_sum(tf.square(diff), axis=-1)  # [H,W,C]
        class_label = tf.argmin(pixel_class_diff, axis=-1)  # [H,W]
    return class_label


def class_to_label_image(class_label, classes_color_values):
    return tf.gather(classes_color_values, class_label)


def rotate_crop(img, rotation, crop=True, interpolation='NEAREST'):
    with tf.name_scope('RotateCrop'):
        rotated_image = tf_rotate(img, rotation, interpolation)
        if crop:
            rotation = tf.abs(rotation)
            original_shape = tf.shape(rotated_image)[:2]
            h, w = original_shape[0], original_shape[1]
            # see https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders for formulae
            old_l, old_s = tf.cond(h > w, lambda: [h, w], lambda: [w, h])
            old_l, old_s = tf.cast(old_l, tf.float32), tf.cast(old_s, tf.float32)
            new_l = (old_l*tf.cos(rotation)-old_s*tf.sin(rotation))/tf.cos(2*rotation)
            new_s = (old_s-tf.sin(rotation)*new_l)/tf.cos(rotation)
            new_h, new_w = tf.cond(h > w, lambda: [new_l, new_s], lambda: [new_s, new_l])
            new_h, new_w = tf.cast(new_h, tf.int32), tf.cast(new_w, tf.int32)
            bb_begin = tf.cast(tf.ceil((h-new_h)/2), tf.int32), tf.cast(tf.ceil((w-new_w)/2), tf.int32)
            rotated_image = rotated_image[bb_begin[0]:h-bb_begin[0], bb_begin[1]:w-bb_begin[1], :]
        return rotated_image