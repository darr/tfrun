#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : softmaxGraph.py
# Create date : 2018-03-04 20:58
# Modified date : 2018-03-05 16:08
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

class SoftmaxGraph(BaseObject):

    def __init__(self):
        super(SoftmaxGraph, self).__init__()
        self._graph_path = "./"
        self._graph_name = self.__class__.__name__
        self._graph = None

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
            x = tf.placeholder(tf.float32, [None, 784],name="placeholder_x")
            w = tf.Variable(tf.zeros([784, 10]),name="variable_w")
            b = tf.Variable(tf.zeros([10]),name="variable_b")
            y = tf.add(tf.matmul(x, w, name="mymatmul") ,b,name="predict_add")
            pybase.pylog.info(y)
            y_ = tf.placeholder(tf.int64, [None],name="placeholder_y_")
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
            tf.summary.scalar('cross_entropy', cross_entropy)
            train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy,name="train_step")
            correct_prediction = tf.equal(tf.argmax(y, 1), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")
            tf.summary.scalar('accuracy', accuracy)
            validation_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="validation_accuracy")
            tf.summary.scalar('validation_accuracy', validation_accuracy)
            test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="test_accuracy")
            tf.summary.scalar('test_accuracy', test_accuracy)
            summary = tf.summary.merge_all()
            pybase.pylog.info(summary)

            var_init = tf.global_variables_initializer()

            self._graph = g
            self.write_graph_to_file()

            g.finalize()
            return g

    def create_graph(self):
        g = self.read_graph()
        if g is None:
            g = self.write_graph()
        return g

    def write_graph_to_file(self):
        if not os.path.isdir(self._graph_path):
            os.makedirs(self._graph_path)
        tf.train.write_graph(self._graph.as_graph_def(),self._graph_path,self._graph_name+'.pb',False)

    def read_graph_file(self):
        f = gfile.FastGFile(self._graph_path+self._graph_name+'.pb','rb')
        return f
