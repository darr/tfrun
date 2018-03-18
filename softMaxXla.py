#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : softMaxXla.py
# Create date : 2018-02-14 20:14
# Modified date : 2018-03-03 16:35
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline

import argparse
import os
import time
import tensorflow as tf
import datasets_mnist

from pybase.pyobject import BaseObject
import pybase.pylog

class SoftMaxXla(BaseObject):

    def __init__(self):
        super(SoftMaxXla, self).__init__()

        self.FLAGS = None
        self.train = None
        self.validation = None
        self.test = None
        self.images_placeholder = None
        self.labels_placeholder = None
        self.saver = None
        self.checkpoint_file = None
        self.logits = None
        self.global_step = None
        self.DIR_NAME = "/softMaxXla"
        #self.sess = tf.Session()

    def run_softmax_xla(self):
        mnist_data = input_data.read_data_sets(self.FLAGS.input_data_dir)

        x = tf.placeholder(tf.float32, [None, 784],name="placeholder_x")
        w = tf.Variable(tf.zeros([784, 10]),name="variable_w")
        b = tf.Variable(tf.zeros([10]),name="variable_b")
        y = tf.matmul(x, w) + b

        y_ = tf.placeholder(tf.int64, [None],name="placeholder_y_")

        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        config = tf.ConfigProto()
        jit_level = 0
        if self.FLAGS.xla:
            jit_level = tf.OptimizerOptions.ON_1

        print(config)
        config.graph_options.optimizer_options.global_jit_level = jit_level
        print(config)

        run_metadata = tf.RunMetadata()
        print(run_metadata)
        sess = tf.Session(config=config)
        print(sess)
        tf.global_variables_initializer().run(session=sess)

        train_loops = 500
        for i in range(train_loops):
            print("%d" % i)
            batch_xs, batch_ys = mnist_data.train.next_batch(100)

            if i == train_loops - 1:
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                with open('timeline.ctf.json', 'w') as trace_file:
                    trace_file.write(trace.generate_chrome_trace_format())
            else:
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: mnist_data.test.images, y_: mnist_data.test.labels}))
        sess.close()
