import os

import tensorflow as tf
from Model.model import inference


class Model:
    def __init__(self, cekpoint_path, filter_size, skip, num_clases):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        with tf.Graph().as_default():
            self.images_placeholder, self.labels_placeholder = self.placeholder_inputs(1)
            self.logits = inference(images=self.images_placeholder, filter_size=filter_size, skip=skip,
                                    num_clases=num_clases)
            self.prediction = model.predictions(logits=self.logits)
            self.saver = tf.train.Saver()
            self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
            self.saver.restore(self.sess, cekpoint_path)

    @staticmethod
    def placeholder_inputs(batch_size):
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 688, 1024, 3))
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size, 688, 1024))
        return images_placeholder, labels_placeholder

    def gen_prediction(self, images):
        pps = []
        for i in range(len(images)):
            batch_x = images[i]
            pred = self.sess.run(self.prediction, feed_dict={self.images_placeholder: batch_x.reshape(1, 688, 1024, 3)})
            pps.append(pred.reshape(688, 1024))
        return pps