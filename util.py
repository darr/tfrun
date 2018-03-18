#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : util.py
# Create date : 2018-02-08 18:42
# Modified date : 2018-02-13 19:46
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################

import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

def getBatchFeedDict(mnist_data,FLAGS,x,y_):
    batch_xs,batch_ys = mnist_data.train.next_batch(FLAGS.batch_size)
    full_feed_dict = {x:batch_xs,y_:batch_ys}
    return full_feed_dict

def getInputData(input_data_dir):
    mnist = input_data.read_data_sets(input_data_dir,one_hot=True)
    return mnist

def checkDir(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    return ckpt_dir

def writeGraph(sess):
    tf.train.write_graph(sess.graph_def,'./tmp/tfmodel','train.pbtxt')

def sessionRun(saver,sess,ckpt_dir):
    checkDir(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
    return sess

def getSession(saver,ckpt_dir):
    sess = tf.Session()
    return sessionRun(saver,sess,ckpt_dir)

def getSessionWithGraph(saver,graph,ckpt_dir):
    sess = tf.Session(target="",graph=graph,config=None)
    return sessionRun(saver,sess,ckpt_dir)

def saveModel(saver,sess,global_step,ckpt_dir):
    checkDir(ckpt_dir)
    saver.save(sess,ckpt_dir+"/model.ckpt",global_step=global_step)

