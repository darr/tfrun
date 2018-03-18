#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : dealMnist.py
# Create date : 2018-02-14 20:14
# Modified date : 2018-03-18 12:27
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
import softmaxGraph
import cnnGraph
import datasetsMnist
#import util

DIR_NAME = "/dealmnist"

def write_tfrecords():
    data_sets = input_data.read_data_sets("MNIST_data/",dtype=tf.uint8,reshape=False,validation_size=5000)
    convert_to(data_sets.train,'train')
    convert_to(data_sets.validation,'validation')
    convert_to(data_sets.test,'test')

    #mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(data_set,name):
    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples

    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' % (images.shape[0],num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]
    filename = os.path.join("MNIST_data/",name+".tfrecords")
    print("Writint",filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
                                    'height':_int64_feature(rows),
                                    'width':_int64_feature(cols),
                                    'depth':_int64_feature(depth),
                                    'label':_int64_feature(int(labels[index])),
                                    'image_raw':_bytes_feature(image_raw)
        }))

        writer.write(example.SerializeToString())

    writer.close()

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
                                        serialized_example,
                                        features={
                                            'image_raw':tf.FixedLenFeature([],tf.string),
                                            'label':tf.FixedLenFeature([],tf.int64),})
    image = tf.decode_raw(features['image_raw'],tf.uint8)
    print(image.shape)
    image.set_shape([784])
    print(image.shape)
    print(image)
    image = tf.cast(image,tf.float32)*(1./255) - 0.5
    label = tf.cast(features['label'],tf.int32)
    return image,label

def inputs(train,batch_size,num_epochs):
    if not num_epochs: num_epochs = None
    TRAIN_FILE = "train.tfrecords"
    VALIDATION_FILE = "validation.tfrecords"
    TRAIN_DIR = "./MNIST_data/"
    filename = os.path.join(TRAIN_DIR,TRAIN_FILE if train else VALIDATION_FILE)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename],num_epochs=num_epochs)
        image,label = read_and_decode(filename_queue)
        print(image)
        print(label)
        images,sparse_labels = tf.train.shuffle_batch(
            [image,label],batch_size=batch_size,num_threads=2,
            capacity = 1000+3*batch_size,
            min_after_dequeue=1000)
    return images, sparse_labels

def checkDir(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    return ckpt_dir

def sessionRun(saver,sess,ckpt_dir):
    checkDir(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        #init = tf.global_variables_initializer()
        #sess.run(init)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        pass
        #init = tf.global_variables_initializer()
        #sess.run(init)
    return sess

class DealMnist(BaseObject):

    def __init__(self):
        super(DealMnist, self).__init__()

    @property
    def FLAGS(self):
        return self._FLAGS

    @FLAGS.setter
    def FLAGS(self,value):
        self._FLAGS = value

    def run_mnist(self):
        self.run_softmax()
        #self.run_cnn()

    def run_softmax(self):
        train ,validation,test = datasetsMnist.read_data_sets(self.FLAGS.input_data_dir, self.FLAGS.fake_data)
        smxGraph = softmaxGraph.SoftmaxGraph()
        smxGraph.graph_path = "./graph/"

        g = smxGraph.graph

        config = tf.ConfigProto()
        jit_level = 0
        if self.FLAGS.xla:
            jit_level = tf.OptimizerOptions.ON_1

        print(config)
        config.graph_options.optimizer_options.global_jit_level = jit_level
        print(config)
        run_metadata = tf.RunMetadata()
        print(run_metadata)
        sess = tf.Session(config=config,graph=g)

        x = sess.graph.get_tensor_by_name("placeholder_x:0")
        y_ = sess.graph.get_tensor_by_name("placeholder_y_:0")
        y = sess.graph.get_operation_by_name("predict_add")
        train_step = sess.graph.get_operation_by_name("train_step")
        init = sess.graph.get_operation_by_name("init")
        summary = sess.graph.get_tensor_by_name("Merge/MergeSummary:0")
        summary_writer = tf.summary.FileWriter(self.FLAGS.log_dir, sess.graph)
        sess.run(init)

        train_loops = 500
        for i in range(train_loops):
            print("%d" % i)
            #batch_xs, batch_ys = mnist_data.train.next_batch(100)
            batch_xs, batch_ys = train.next_batch(100)

            if i == train_loops - 1:
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                with open('timeline.ctf.json', 'w') as trace_file:
                    trace_file.write(trace.generate_chrome_trace_format())
            else:
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            validation_accuracy_tensor = sess.graph.get_tensor_by_name("validation_accuracy:0")
            validation_accuracy = sess.run(validation_accuracy_tensor, feed_dict={x: validation.images, y_: validation.labels})
            print("validation accuracy:%s" % validation_accuracy)

            summary_str = sess.run(summary, feed_dict={x: batch_xs,y_:batch_ys})
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

        c1_tensor = sess.graph.get_tensor_by_name("test_accuracy:0")
        print(sess.run(c1_tensor, feed_dict={x: test.images, y_: test.labels}))
        sess.close()

    def do_eval(self,mnist_data,graph,saver):
        sess = tf.Session(graph=graph)
        sess = sessionRun(saver,sess,self.FLAGS.ckpt_dir+DIR_NAME)
        x = sess.graph.get_tensor_by_name("placeholder_x:0")
        y_ = sess.graph.get_tensor_by_name("placeholder_y_:0")
        keep_prob = sess.graph.get_tensor_by_name("dropout/Placeholder:0")
        steps = sess.graph.get_tensor_by_name("steps:0")
        c1_tensor = sess.graph.get_tensor_by_name("accuracy_1:0")
        test_accuracy = sess.run(c1_tensor, feed_dict={x: mnist_data.test.images, y_: mnist_data.test.labels,keep_prob:1.0,steps:0})
        pybase.pylog.info("test_accuracy:%s" % test_accuracy)
        sess.close()
        return test_accuracy

    def run_cnn(self):
        mnist_data = util.getInputData(self.FLAGS.input_data_dir)
        #mnist_data = input_data.read_data_sets(self.FLAGS.input_data_dir)
        smxGraph = cnnGraph.CnnGraph()
        smxGraph.graph_path = "./graph/"

        g = smxGraph.graph
        saver = smxGraph.saver

        config = tf.ConfigProto()
        jit_level = 0
        if self.FLAGS.xla:
            jit_level = tf.OptimizerOptions.ON_1

        print(config)
        config.graph_options.optimizer_options.global_jit_level = jit_level
        print(config)
        run_metadata = tf.RunMetadata()
        print(run_metadata)
        sess = tf.Session(config=config,graph=g)

        x = sess.graph.get_tensor_by_name("placeholder_x:0")
        y_ = sess.graph.get_tensor_by_name("placeholder_y_:0")
        y = sess.graph.get_tensor_by_name("fc2/predict_add:0")
        train_step = sess.graph.get_operation_by_name("train_step")
        keep_prob = sess.graph.get_tensor_by_name("dropout/Placeholder:0")
        steps = sess.graph.get_tensor_by_name("steps:0")
        cur_step= sess.graph.get_tensor_by_name("cur_step:0")
        test_accuracy_placeholder = sess.graph.get_tensor_by_name("test_accuracy_placeholder:0")
        test_accuracy_assign = sess.graph.get_operation_by_name("test_accuracy_assign")
        global_step= sess.graph.get_tensor_by_name("global_step:0")
        pybase.pylog.info(global_step)
        init = sess.graph.get_operation_by_name("init")
        summary = sess.graph.get_tensor_by_name("Merge/MergeSummary:0")

        summary_writer = tf.summary.FileWriter(self.FLAGS.log_dir, sess.graph)
        sess.run(init)
        sess = sessionRun(saver,sess,self.FLAGS.ckpt_dir+DIR_NAME)
        start = sess.run(cur_step)
        pybase.pylog.error(start)
        end = self.FLAGS.max_steps

        for i in range(start,end):
            print("%d" % i)
            batch_xs, batch_ys = mnist_data.train.next_batch(100)

            if i == end - 1:
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,keep_prob:0.5,steps:i}, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                with open('timeline.ctf.json', 'w') as trace_file:
                    trace_file.write(trace.generate_chrome_trace_format())
            else:
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,keep_prob:0.5,steps:i})

            gs = sess.run(global_step,feed_dict={steps:i})
            cs = sess.run(cur_step)

            if i > 0 and i % 100 == 0:
                util.saveModel(saver,sess,i,self.FLAGS.ckpt_dir+DIR_NAME)
                summary_str = sess.run(summary, feed_dict={x: batch_xs,y_:batch_ys,keep_prob:0.5,steps:i})
                summary_writer.add_summary(summary_str, i)
                summary_writer.flush()
                t_accuracy = self.do_eval(mnist_data,g,saver)
                sess.run(test_accuracy_assign,feed_dict={test_accuracy_placeholder:t_accuracy})

        sess.close()

