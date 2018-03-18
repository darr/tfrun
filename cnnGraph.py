#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : cnnGraph.py
# Create date : 2018-03-04 20:58
# Modified date : 2018-03-06 15:11
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
import pybase.pylog

def deepnn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x,[-1,28,28,1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5,5,1,32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5,5,32,64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7*7*64,1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024,10])
        b_fc2 = bias_variable([10])
        #y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
        y_conv = tf.add(tf.matmul(h_fc1_drop,W_fc2), b_fc2, name="predict_add")

    return y_conv,keep_prob

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

class CnnGraph(BaseObject):

    def __init__(self):
        super(CnnGraph, self).__init__()
        self._graph_path = "./"
        self._graph_name = self.__class__.__name__
        self._graph = None
        self._saver = None

    @property
    def saver(self):
        if self._saver == None:
            return None
        else:
            return self._saver

    @property
    def graph_path(self):
        return self._graph_path

    @graph_path.setter
    def graph_path(self,value):
        if isinstance(value,basestring):
            self._graph_path = value
        else:
            pybase.pylog.warning("graph path is not a string")

    @property
    def graph(self):
        if self._graph == None:
            return self.create_graph()
        else:
            return self._graph


    def read_graph(self):
        try:
            f =  self.read_graph_file()
            con = f.read()
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(con)
            tf.import_graph_def(graph_def,name='')
            g = tf.get_default_graph()
            self._graph = g
            return g
        except:
            return None

    def write_graph(self):
        g = tf.Graph()

        with g.as_default():
            x= tf.placeholder(tf.float32,[None,784],name="placeholder_x")
            y_ = tf.placeholder(tf.float32,[None,10],name="placeholder_y_")
            steps = tf.placeholder(tf.int32,name="steps")
            y_conv,keep_prob = deepnn(x)
            with tf.name_scope('loss'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)

            cross_entropy = tf.reduce_mean(cross_entropy,name="cross_entropy")
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,name="train_step")

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
                correct_prediction = tf.cast(correct_prediction,tf.float32)

            accuracy = tf.reduce_mean(correct_prediction,name="accuracy")
            tf.summary.scalar('accuracy', accuracy)
            cur_step = tf.Variable(0,name='cur_step')
            global_step = tf.assign(cur_step,steps,name="global_step")

            test_accuracy_placeholder = tf.placeholder(tf.float32,name="test_accuracy_placeholder")
            test_accuracy = tf.Variable(0.0,name="test_accuracy")
            tf.assign(test_accuracy,test_accuracy_placeholder,name="test_accuracy_assign")
            tf.summary.scalar('test_accuracy', test_accuracy)

            summary = tf.summary.merge_all()

            var_init = tf.global_variables_initializer()
            pybase.pylog.info(var_init)

            self._saver = tf.train.Saver()

            self._graph = g
            self.write_graph_to_file()

            g.finalize()
            return g

    def create_graph(self):
        g = self.write_graph()
        return g
#       g = self.read_graph()
#       if g is None:
#           g = self.write_graph()
#       return g

    def write_graph_to_file(self):
        if not os.path.isdir(self._graph_path):
            os.makedirs(self._graph_path)
        tf.train.write_graph(self._graph.as_graph_def(),self._graph_path,self._graph_name+'.pb',False)

    def read_graph_file(self):
        f = gfile.FastGFile(self._graph_path+self._graph_name+'.pb','rb')
        return f
