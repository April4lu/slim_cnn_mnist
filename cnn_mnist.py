# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data 

mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)
def CNN(input,is_training):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
         # 该参数有一个默认值，ops.GraphKeys.UPDATE_OPS，当取默认值时，slim会在当前批训练完成后再更新均
         # 值和方差，这样会存在一个问题，就是当前批数据使用的均值和方差总是慢一拍，最后导致训练出来的模
         # 型性能较差。所以，一般需要将该值设为None，这样slim进行批处理时，会对均值和方差进行即时更新，
         # 批处理使用的就是最新的均值和方差    
    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),             
                        normalizer_fn = slim.batch_norm,
                        normalizer_params = batch_norm_params
            ):
        net  = slim.conv2d()
