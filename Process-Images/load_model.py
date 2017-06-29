import os

import model
import tensorflow as tf
from model import inference
from threading import BoundedSemaphore


class Model:
    def __init__(self, checkpoint_path, filter_size, skip, num_clases, gpu_list="0"):
        self.pool_sema = BoundedSemaphore(value=2)
        with tf.Graph().as_default():
            self.images_placeholder, self.labels_placeholder = self.placeholder_inputs(1)
            self.logits = inference(images=self.images_placeholder, filter_size=filter_size, skip=skip,
                                    num_clases=num_clases)
            self.prediction = model.predictions(logits=self.logits)
            self.saver = tf.train.Saver()
            self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4, visible_device_list=gpu_list)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
            self.saver.restore(self.sess, checkpoint_path)

    @staticmethod
    def placeholder_inputs(batch_size):
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 688, 1024, 3))
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size, 688, 1024))
        return images_placeholder, labels_placeholder

    def gen_prediction(self, images):
        with self.pool_sema:
            pps = []
            for i in range(len(images)):
                batch_x = images[i]
                pred = self.sess.run(self.prediction, feed_dict={self.images_placeholder: batch_x.reshape(1, 688, 1024, 3)})
                pps.append(pred.reshape(688, 1024))
            return pps
