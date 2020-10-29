# coding=utf-8

import tensorflow as tf

if tf.__version__[0] == "1":
    tf.enable_eager_execution()

import tf_kge.data as data
import tf_kge.dataset as dataset
import tf_kge.utils as utils

from tf_kge.data import *