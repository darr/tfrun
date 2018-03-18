#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : runMnist.py
# Create date : 2018-03-04 20:58
# Modified date : 2018-03-18 12:18
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
from pybase.pyobject import BaseObject
from tensorflow.python.platform import gfile
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.python import debug as tf_debug
import pybase.pylog
import time

DIR_NAME = "/runmnist"

def saveModel(saver,sess,global_step,ckpt_dir):
    checkDir(ckpt_dir)
    saver.save(sess,ckpt_dir+"/model.ckpt",global_step=global_step)

def sessionRun(saver,sess,ckpt_dir):
    checkDir(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        pass
    return sess

def checkDir(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    return ckpt_dir

def write_tfrecords():
    data_sets = input_data.read_data_sets("MNIST_data/",dtype=tf.uint8,reshape=False,validation_size=5000)
    convert_to(data_sets.train,'train')
    convert_to(data_sets.validation,'validation')
    convert_to(data_sets.test,'test')

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
    image.set_shape([784])
    image = tf.cast(image,tf.float32)*(1./255) - 0.5
    label = tf.cast(features['label'],tf.int32)
    return image,label

def inputs(train,batch_size,num_epochs):
    if not num_epochs: num_epochs = None
    TRAIN_FILE = "train.tfrecords"
    VALIDATION_FILE = "validation.tfrecords"
    TEST_FILE = "test.tfrecords"
    TRAIN_DIR = "./MNIST_data/"
    filename = os.path.join(TRAIN_DIR,TRAIN_FILE if train is not None else VALIDATION_FILE)
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

class RunMnist(BaseObject):

    def __init__(self):
        super(RunMnist, self).__init__()
        self._graph_path = "./tmp/%s/" % self.__class__.__name__
        self._graph_name = self.__class__.__name__
        self._graph = None

    def write_graph_to_file(self):
        if not os.path.isdir(self._graph_path):
            os.makedirs(self._graph_path)
        tf.train.write_graph(self._graph.as_graph_def(),self._graph_path,self._graph_name+'.pb',False)

    def read_graph_file(self):
        f = gfile.FastGFile(self._graph_path+self._graph_name+'.pb','rb')
        return f

    def run_train(self):
        with tf.Graph().as_default():
            isTrain = tf.placeholder(tf.bool,name="isTrain")
            images,labels = inputs(train=isTrain,batch_size=100,num_epochs=500)
            logits = mnist.inference(images,128,32)
            loss = mnist.loss(logits,labels)
            tf.summary.scalar('loss', loss)
            train_op = mnist.training(loss,0.01)
            evaluation = mnist.evaluation(logits,labels)
            tf.summary.scalar('evaluation', evaluation)
            cur_step = tf.Variable(0,name='cur_step')
            summary = tf.summary.merge_all()
            init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
            saver = tf.train.Saver()

            sess = tf.Session()
            self._graph = sess.graph
            sess.run(init_op)
            self.write_graph_to_file()
            sess = sessionRun(saver,sess,self.FLAGS.ckpt_dir+DIR_NAME)

            if self.FLAGS.debug and self.FLAGS.tensorboard_debug_address:
                raise ValueError( "The --debug and --tensorboard_debug_address flags are mutually " "exclusive.")
            if self.FLAGS.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type=self.FLAGS.ui_type)
            elif self.FLAGS.tensorboard_debug_address:
                sess = tf_debug.TensorBoardDebugWrapperSession( sess, self.FLAGS.tensorboard_debug_address)

            summary_writer = tf.summary.FileWriter(self.FLAGS.log_dir, sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            start = sess.run(cur_step)

            try:
                step = start
                while not coord.should_stop():
                    start_time = time.time()
                    _,loss_value,prd,summary_str = sess.run([train_op,loss,evaluation,summary],feed_dict={isTrain:True})
                    summary_writer.add_summary(summary_str,step)
                    summary_writer.flush()
                    print(step)
                    duration = time.time() - start_time

                    if step % 100 == 0:
                        cs = sess.run(cur_step.assign(step))
                        saveModel(saver,sess,step,self.FLAGS.ckpt_dir+DIR_NAME)
                        print("Step :%d: loss=%.2f (%.3f sec) evaluation:%s" % (step,loss_value,duration,prd))
                    step += 1

                    if step > 3000:
                        prd = sess.run(evaluation,feed_dict={isTrain:False})
                        print("loss value:%s evaluation:%s" % (loss_value,prd))
                        break
            except tf.errors.OutOfRangeError:
                print("Done training for %d epochs, %d steps." % (1000,step))
            finally:
                coord.request_stop()
            coord.join(threads)
            sess.close()


