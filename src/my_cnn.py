#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN model(eager execution)
"""
__author__ = 'Taiki Horiuchi'
__version__ = '1.0'
__date__    = "2019/06/26 12:28"


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.nn import softmax_cross_entropy_with_logits
from tensorflow.train import AdamOptimizer
import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()


class my_cnn(object):
    """
    CNNモデル
    """
    
    def __init__(self):
        """
        各層のフィルタを設定
        """
        self.conv1 = Conv2D(filters=32, kernel_size=[5,5], padding='same', activation='relu', name='conv1')
        self.pool1 = MaxPool2D(pool_size=[2,2], strides=[2,2], name='pool1')
        self.conv2 = Conv2D(filters=64, kernel_size=[5,5], padding='same', activation='relu', name='conv2')
        self.pool2 = MaxPool2D(pool_size=[2,2], strides=[2,2], name='pool2')
        self.flatten = Flatten(name='flatten')
        self.fc1 = Dense(units=1024, activation='relu', name='dense1')
        self.dropout = Dropout(rate=0.5)
        self.fc2 = Dense(units=10, activation=None, name='dense2')
        
    def call(self, batch_img):
        """
        フィルタ同士を繋げ，ネットワークを生成する
        """
        feature_map = self.conv1(batch_img)
        feature_map = self.pool1(feature_map)
        feature_map = self.conv2(feature_map)
        feature_map = self.pool2(feature_map)
        feature_map = self.flatten(feature_map)
        feature_map = self.fc1(feature_map)
        feature_map = self.dropout(feature_map)
        feature_map = self.fc2(feature_map)
        return feature_map
        
