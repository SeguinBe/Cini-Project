import tensorflow as tf
from . import upsample


def model_fn(mode, features, labels, params):

    filter_size = params['filter_size']
    skip = params['skip']
    num_classes = params['num_classes']

    logits = inference(features['images'], filter_size, skip, num_classes)
    prediction_probs = tf.nn.softmax(logits)
    prediction_labels = tf.argmax(logits, axis=-1)

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        onehot_labels = tf.one_hot(indices=labels, depth=num_classes)

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels),
                                  name="loss")
    else:
        loss = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        ema = tf.train.ExponentialMovingAverage(0.9)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema.apply([loss]))
        ema_loss = ema.average(loss)
        tf.summary.scalar('losses/loss_EMA', ema_loss)
        tf.summary.scalar('losses/loss_batch', loss)

        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
    else:
        ema_loss, train_op = None, None

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {'accuracy': tf.metrics.accuracy(labels, predictions=prediction_labels)}
    else:
        metrics = None

    return tf.estimator.EstimatorSpec(mode,
                                      predictions={'probs': prediction_probs,
                                                   'labels': prediction_labels},
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=metrics
                                      )


def inference(images, filter_size, skip, num_classes):

    """Model function for CNN.
    Args:
        :param images: Images placeholder, from inputs().
        :param filter_size: Size of the convolution kernel (One number or list of length 17)
        :param skip: Number of skip connections (Number between 0 and 3)

    Returns:
        :return softmax_linear: Output tensor with the computed logits.
    """

    #tf.summary.image('input', images, 3)

    if type(filter_size) == int:
        filter_size = [filter_size] * 17

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
        filters=128,
        kernel_size=[filter_size[6], filter_size[6]],
        padding="same",
        activation=tf.nn.relu,
        name="conv4_1")

    conv4_2 = tf.layers.conv2d(
        inputs=conv4_1,
        filters=128,
        kernel_size=[filter_size[7], filter_size[7]],
        padding="same",
        activation=tf.nn.relu,
        name="conv4_2")

    pool4 = tf.layers.max_pooling2d(inputs=conv4_2, pool_size=[2, 2], strides=2, name="pool_4")

    up_sample_1 = upsample.upsample_layer(inputs=pool4, channels_in=128, channels_out=128, factor=2, name="up_sample_1")

    if skip == 3:
        skip_1 = tf.concat([up_sample_1, pool3], 3)
    else:
        skip_1 = up_sample_1

    # Convolution block #5
    conv5_1 = tf.layers.conv2d(
        inputs=skip_1,
        filters=128,
        kernel_size=[filter_size[8], filter_size[8]],
        padding="same",
        activation=tf.nn.relu,
        name="conv5_1")

    conv5_2 = tf.layers.conv2d(
        inputs=conv5_1,
        filters=128,
        kernel_size=[filter_size[9], filter_size[9]],
        padding="same",
        activation=tf.nn.relu,
        name="conv5_2")

    up_sample_2 = upsample.upsample_layer(inputs=conv5_2, channels_in=128, channels_out=128, factor=2, name="up_sample_2")

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
        filters=num_classes,
        kernel_size=[filter_size[16], filter_size[16]],
        padding="same",
        name="conv8_3")

    logits = conv8_3

    return logits
