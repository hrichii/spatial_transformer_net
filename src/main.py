#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial transformer net
"""
#import math
import numpy as np
import tensorflow as tf
#from tensorflow.keras.datasets.mnist import load_data as tfload
import tensorflow.contrib.eager as tfe
#import matplotlib.pyplot as plt
from my_cnn import my_cnn
from my_stn import my_stn
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
from tensorflow.nn import softmax_cross_entropy_with_logits
from tensorflow.train import AdamOptimizer

tfe.enable_eager_execution()
#
#INPATH = "../data/"
#filepath=INPATH+"FILENAME.png"
#H = 28
#W = 28
#
#def write_gray2png(filepath, gray):
#    """
#    グレースケールの配列をpngに変換し，保存
#    """
#    H = gray.shape[0]
#    W = gray.shape[1]
#    binary = tf.image.encode_png(gray.reshape(H,W,1))
#    tf.write_file(filepath, binary)
#
#
##1 load data images(10000x28x28)
#(x_train, y_train), (x_test, y_test) = tfload()
#x = x_train[0,:,:].reshape(H,W,1)

def loss(itr,batch_img,batch_label,model):
    
    batch_pred = model.call(batch_img)
    #誤差の計算
    loss = softmax_cross_entropy_with_logits(labels = batch_label, logits = batch_pred)
    loss = tf.reduce_mean(loss)
    
    #100イテレーション毎に誤差を出力
    if (itr+1) % 100 == 0:
        # 计算准确率
        predict = tf.argmax(batch_pred, 1).numpy()
        target = np.argmax(batch_label, 1)
        accuracy = np.sum(predict == target)/len(target)

        print("step {}:\tloss = {}\taccuracy = {}".format(itr+1, loss.numpy(), accuracy))

    return loss


if __name__ == '__main__':
    height =28
    width = 28
    num_batch =200


    #1 データ読み込み
    mnist = read_data_sets("/data/mnist", one_hot=True)
    
    #2 モデル生成
    model = my_stn()
    #model = my_cnn()
    
    #3 訓練
    optimizer = AdamOptimizer(learning_rate=1e-3)
    batch_size = 200
    for itr in range(1000):
        #バッチ毎に分ける
        batch_img, batch_label = mnist.train.next_batch(batch_size)
        batch_img = batch_img.reshape([-1,28,28,1])
    
        #誤差の最小化
        optimizer.minimize(lambda: loss(itr,batch_img,batch_label,model))