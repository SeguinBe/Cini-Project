import time

import model
import tensorflow as tf
from model import inference, evaluation, training
from utils import batch_iter


def placeholder_inputs(batch_size, image_size):

    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, image_size[0], image_size[1], image_size[2]))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size, image_size[0], image_size[1]))
    return images_placeholder, labels_placeholder

def run_training(images, labels, filter_size, skip, learning_rate, batch_size, epochs, num_clases, gpu_memory_fraction, file_writer, model_path):

    with tf.Graph().as_default():

        images_placeholder, labels_placeholder = placeholder_inputs(batch_size, images[0].shape)

        logits = inference(images=images_placeholder, filter_size=filter_size, skip=skip, num_clases=num_clases)

        loss = model.loss(logits, labels_placeholder, num_clases=num_clases)

        train_op = training(loss, learning_rate)

        eval_correct = evaluation(logits, labels_placeholder, num_clases=num_clases)

        summary = tf.summary.merge_all()

        saver = tf.train.Saver()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter(file_writer)
            writer.add_graph(sess.graph)

            i = 0
            for batch_y, batch_x in batch_iter(labels, images, batch_size, epochs=epochs):
                [_, train_loss] = sess.run([train_op, loss], feed_dict={images_placeholder: batch_x, labels_placeholder: batch_y})
                if i % 5 == 0:
                    train_accuracy = sess.run(eval_correct, feed_dict={images_placeholder: batch_x, labels_placeholder: batch_y})
                    print("step %d, train accuracy %g train loss %g" % (i, train_accuracy, train_loss))
                    summary_str = sess.run(summary, feed_dict={images_placeholder: batch_x, labels_placeholder: batch_y})
                    writer.add_summary(summary_str, i)
                    writer.flush()
                i += 1

            saver.save(sess, model_path)

def get_testAccuracy(images, labels, filter_size, skip, gpu_memory_fraction, model_path):

    with tf.Graph().as_default():

        images_placeholder, labels_placeholder = placeholder_inputs(1, images[0].shape)

        logits = inference(images=images_placeholder, filter_size=filter_size, skip=skip)

        eval_correct = evaluation(logits, labels_placeholder)

        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            saver.restore(sess, model_path)
            start_time = time.time()
            pps = []
            for i in range(len(images)):
                batch_x = images[i]
                batch_y = labels[i]
                pred = sess.run(eval_correct, feed_dict={images_placeholder: batch_x.reshape(1, images[0].shape[0], images[0].shape[1], images[0].shape[2]), labels_placeholder: batch_y.reshape(1, images[0].shape[0], images[0].shape[1])})
                pps.append(pred)
            duration = time.time() - start_time
            print(duration)
            return pps


def gen_prediction(images, filter_size, skip, num_clases, gpu_memory_fraction, model_path):

    with tf.Graph().as_default():

        images_placeholder, labels_placeholder = placeholder_inputs(1, images[0].shape)

        logits = inference(images=images_placeholder, filter_size=filter_size, skip=skip, num_clases=num_clases)
        prediction = model.predictions(logits=logits)

        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            saver.restore(sess, model_path)
            start_time = time.time()
            pps = []
            for i in range(len(images)):
                batch_x = images[i]
                pred = sess.run(prediction, feed_dict={images_placeholder: batch_x.reshape(1,images[0].shape[0], images[0].shape[1], images[0].shape[2])})
                pps.append(pred.reshape(images[0].shape[0], images[0].shape[1]))
            duration = time.time() - start_time
            print(duration)
            return pps
