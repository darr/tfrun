#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : appargs.py
# Create date : 2018-02-08 23:30
# Modified date : 2018-03-08 18:29
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################

import argparse
import os
import etc

def getArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument(
	  '--learning_rate',
	  type=float,
	  default=etc.LEARNING_RATE,
	  help='Initial learning rate.'
	)
	parser.add_argument(
	  '--keep_prob',
	  type=float,
	  default=etc.KEEP_PROB,
	  help='keep prob.'
	)
	parser.add_argument(
	  '--max_steps',
	  type=int,
	  default=etc.MAX_STEPS,
	  help='Number of steps to run trainer.'
	)
	parser.add_argument(
	  '--print_step',
	  type=int,
	  default=etc.PRINT_STEP,
	  help='print step'
	)
	parser.add_argument(
	  '--batch_size',
	  type=int,
	  default=etc.BATCH_SIZE,
	  help='Batch size.  Must divide evenly into the dataset sizes.'
	)
	parser.add_argument(
	  '--ckpt_dir',
	  type=str,
	  default=etc.CKPT_DIR,
	  help='ckpt dir'
	)
	parser.add_argument(
	  '--input_data_dir',
	  type=str,
#	  default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/input_data'),
      default=etc.INPUT_DATA_DIR,
	  help='Directory to put the input data.'
	)
	parser.add_argument(
	  '--log_dir',
	  type=str,
#	  default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/logs/fully_connected_feed'),
      default=etc.LOG_DIR,
	  help='Directory to put the log data.'
	)
	parser.add_argument(
	  '--hidden1',
	  type=int,
	  default=128,
	  help='Number of units in hidden layer 1.'
	)
	parser.add_argument(
	  '--hidden2',
	  type=int,
	  default=32,
	  help='Number of units in hidden layer 2.'
	)
	parser.add_argument(
	  '--fake_data',
	  default=False,
	  help='If true, uses fake data for unit testing.',
	  action='store_true'
	)
	parser.add_argument(
	  '--xla',
	  default=True,
	  help='If true, uses fake data for unit testing.',
	  action='store_true'
	)
	parser.add_argument(
	  '--debug',
	  default=False,
	  help='If true, uses tfdbf',
	  action='store_true'
	)
	parser.add_argument(
	  '--tensorboard_debug_address',
      type=str,
	  default=None,
      help="Connect to the TensorBoard Debugger Plugin backend specified by "
      "the gRPC address (e.g., localhost:1234). Mutually exclusive with the "
      "--debug flag.",
	)
	parser.add_argument(
      "--ui_type",
      type=str,
      default="curses",
      help="Command-line user interface type (curses | readline)")

	return parser
