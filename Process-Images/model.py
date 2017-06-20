"""Builds the cardboard extraction network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.
"""

import tensorflow as tf
import upsample


def inference(images, filter_size, skip, num_clases):

    """Model function for CNN.
    Args:
        :param images: Images placeholder, from inputs().
        :param filter_size: Size of the convolution kernel (One number or list of length 17)
        :param skip: Number of skip connections (Number between 0 and 3)
    
    Returns:
        :return softmax_linear: Output tensor with the computed logits.
    """


    tf.summary.image('input', images, 3)

    if type (filter_size) == int:
        filter_size  = [filter_size] * 17

    # Convolution block #1
    conv1_1 = tf.layers.conv2d(
        inputs=images,
        filters=32,
        kernel_size=[filter_size[0], filter_size[0]],
        padding="same",
        activation=tf.nn.relu,
        name="conv1_1")

    conv1_2 = tf.layers.conv2d(
        inputs=conv1_1,
        filters=32,
        kernel_size=[filter_size[1], filter_size[1]],
        padding="same",
        activation=tf.nn.relu,
        name="conv1_2")

    pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2, name="pool_1")

    # Convolution block #2
    conv2_1 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[filter_size[2], filter_size[2]],
        padding="same",
        activation=tf.nn.relu,
        name="conv2_1")

    conv2_2 = tf.layers.conv2d(
        inputs=conv2_1,
        filters=64,
        kernel_size=[filter_size[3], filter_size[3]],
        padding="same",
        activation=tf.nn.relu,
        name="conv2_2")

    pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2, name="pool_2")

    # Convolution block #3
    conv3_1 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[filter_size[4], filter_size[4]],
        padding="same",
        activation=tf.nn.relu,
        name="conv3_1")

    conv3_2 = tf.layers.conv2d(
        inputs=conv3_1,
        filters=128,
        kernel_size=[filter_size[5], filter_size[5]],
        padding="same",
        activation=tf.nn.relu,
        name="conv3_2")

    pool3 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=[2, 2], strides=2, name="pool_3")

    # Convolution block #4
    conv4_1 = tf.layers.conv2d(
        inputs=pool3,
        filters=256,
        kernel_size=[filter_size[6], filter_size[6]],
        padding="same",
        activation=tf.nn.relu,
        name="conv4_1")

    conv4_2 = tf.layers.conv2d(
        inputs=conv4_1,
        filters=256,
        kernel_size=[filter_size[7], filter_size[7]],
        padding="same",
        activation=tf.nn.relu,
        name="conv4_2")

    pool4 = tf.layers.max_pooling2d(inputs=conv4_2, pool_size=[2, 2], strides=2, name="pool_4")

    up_sample_1 = upsample.upsample_layer(inputs=pool4, channels_in=256, channels_out=256, factor=2, name="up_sample_1")

    if skip == 3:
        skip_1 = tf.concat([up_sample_1, pool3], 3)
    else:
        skip_1 = up_sample_1

    # Convolution block #5
    conv5_1 = tf.layers.conv2d(
        inputs=skip_1,
        filters=256,
        kernel_size=[filter_size[8], filter_size[8]],
        padding="same",
        activation=tf.nn.relu,
        name="conv5_1")

    conv5_2 = tf.layers.conv2d(
        inputs=conv5_1,
        filters=256,
        kernel_size=[filter_size[9], filter_size[9]],
        padding="same",
        activation=tf.nn.relu,
        name="conv5_2")

    up_sample_2 = upsample.upsample_layer(inputs=conv5_2, channels_in=256, channels_out=256, factor=2, name="up_sample_2")

    if skip:
        skip_2 = tf.concat([up_sample_2, pool2], 3)
    else:
        skip_2 = up_sample_2

    # Convolution block #6
    conv6_1 = tf.layers.conv2d(
        inputs=skip_2,
        filters=128,
        kernel_size=[filter_size[10], filter_size[10]],
        padding="same",
        activation=tf.nn.relu,
        name="conv6_1")

    conv6_2 = tf.layers.conv2d(
        inputs=conv6_1,
        filters=128,
        kernel_size=[filter_size[11], filter_size[11]],
        padding="same",
        activation=tf.nn.relu,
        name="conv6_2")

    up_sample_3 = upsample.upsample_layer(inputs=conv6_2, channels_in=128, channels_out=128, factor=2, name="up_sample_3")

    if skip >= 2:
        skip_3 = tf.concat([up_sample_3, pool1], 3)
    else:
        skip_3 = up_sample_3

    # Convolution block #7
    conv7_1 = tf.layers.conv2d(
        inputs=skip_3,
        filters=64,
        kernel_size=[filter_size[12], filter_size[12]],
        padding="same",
        activation=tf.nn.relu,
        name="conv7_1")

    conv7_2 = tf.layers.conv2d(
        inputs=conv7_1,
        filters=64,
        kernel_size=[filter_size[13], filter_size[13]],
        padding="same",
        activation=tf.nn.relu,
        name="conv7_2")

    up_sample_4 = upsample.upsample_layer(inputs=conv7_2, channels_in=64, channels_out=64, factor=2, name="up_sample_7")

    if skip >= 1:
        skip_4 = tf.concat([up_sample_4, images], 3)
    else:
        skip_4 = up_sample_4

    # Convolution block #8
    conv8_1 = tf.layers.conv2d(
        inputs=skip_4,
        filters=32,
        kernel_size=[filter_size[14], filter_size[14]],
        padding="same",
        activation=tf.nn.relu,
        name="conv8_1")

    conv8_2 = tf.layers.conv2d(
        inputs=conv8_1,
        filters=32,
        kernel_size=[filter_size[15], filter_size[15]],
        padding="same",
        activation=tf.nn.relu,
        name="conv8_2")

    conv8_3 = tf.layers.conv2d(
        inputs=conv8_2,
        filters=num_clases,
        kernel_size=[filter_size[16], filter_size[16]],
        padding="same",
        name="conv8_3")

    logits = tf.reshape(tensor=conv8_3, shape=(-1, num_clases))

    return logits


def loss(logits, labels, num_clases):

    flat_labels = tf.reshape(tensor=labels, shape=[-1])
    onehot_labels = tf.one_hot(indices=tf.cast(flat_labels, tf.int32), depth=num_clases)

    with tf.name_scope("xent"):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels), name="xent")
        tf.summary.scalar("xent", xent)
        return xent


def training(xent, learning_rate):

    with tf.name_scope("train"):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent, global_step=global_step)
    return train_step


def evaluation(logits, labels, num_clases):

    flat_labels = tf.reshape(tensor=labels, shape=[-1])
    onehot_labels = tf.one_hot(indices=tf.cast(flat_labels, tf.int32), depth=num_clases)

    with tf.name_scope("accuracy"):
        sofmax_logits = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.argmax(sofmax_logits, 1), tf.argmax(onehot_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
        return accuracy

def predictions(logits):
    sofmax_logits = tf.nn.softmax(logits)
    return tf.argmax(sofmax_logits, 1)

