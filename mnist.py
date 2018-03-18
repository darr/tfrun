#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : mnist.py
# Create date : 2018-02-07 10:41
# Modified date : 2018-03-18 12:29
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################

import tensorflow as tf
import sys
import os
import util
import appargs
import dealMnist
import runMnist

FLAGS = None

def test_dealmnist():
    objTest = dealMnist.DealMnist()
    objTest.FLAGS = FLAGS
    objTest.check_Attrs()
    print objTest
    print repr(objTest)
    objTest.run_mnist()

def main(_):

    test_dealmnist()

    #test()
    #runMnist.write_tfrecords()
    objTest = runMnist.RunMnist()
    objTest.FLAGS = FLAGS
    objTest.run_train()

if __name__ == '__main__':
    args = appargs.getArgs()
    FLAGS, unparsed = args.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
