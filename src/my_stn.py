#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN connected STN
"""
__author__ = 'Taiki Horiuchi'
__version__ = '1.0'
__date__    = "2019/06/26 16:08"


#import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from spatial_transformer import transformer
#from tensorflow.nn import softmax_cross_entropy_with_logits
#from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
#from tensorflow.train import AdamOptimizer
#import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()


class my_stn(tf.keras.Model):
    """
    CNN connected STN モデル
    """
    def __init__(self):
        """
        各層のフィルタを設定
        """
        super(my_stn, self).__init__()
        self.flatten1 = Flatten(name='flatten1')
        self.fc1 = Dense(units=20, activation='tanh', name='dense1')
        self.dropout1 = Dropout(rate=0.8, name='dropout1')
        self.fc2 = Dense(units=6, activation='tanh', name='dense2')
        self.conv1 = Conv2D(filters=16, kernel_size=[5,5], strides=[2,2], padding='same', activation='relu', name='conv1')
        self.conv2 = Conv2D(filters=16, kernel_size=[5,5], strides=[2,2], padding='same', activation='relu', name='conv2')
        self.flatten2 = Flatten(name='flatten2')
        self.fc3 = Dense(units=1024, activation='relu', name='dense3')
        self.fc4 = Dense(units=10, activation=None, name='dense4')
        
    def call(self, batch_img):
        """
        フィルタ同士を繋げ，ネットワークを生成する
        """
        feature_map = self.flatten1(batch_img)
        feature_map = self.fc1(feature_map)
        feature_map = self.dropout1(feature_map)
        feature_map = self.fc2(feature_map)
        feature_map = transformer(U=batch_img, theta=feature_map, out_size=(40,40))
#        feature_map = batch_img###############
        feature_map = self.conv1(feature_map)
        feature_map = self.conv2(feature_map)
        feature_map = self.flatten2(feature_map)
        feature_map = self.fc3(feature_map)
        feature_map = self.fc4(feature_map)
        return feature_map
 
    def stn_img(self, batch_img):
        feature_map = self.flatten1(batch_img)
        feature_map = self.fc1(feature_map)
        feature_map = self.dropout1(feature_map)
        feature_map = self.fc2(feature_map)
        feature_map = transformer(U=batch_img, theta=feature_map, out_size=(40,40))
        return feature_map