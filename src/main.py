#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial transformer net
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
#import matplotlib.pyplot as plt
from my_cnn import my_cnn
from my_stn import my_stn
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
from tensorflow.nn import softmax_cross_entropy_with_logits
from tensorflow.train import AdamOptimizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib import summary
import cv2
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


tfk = tf.keras
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

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

#tra_loss_list = []
#tra_acc_list = []
#val_loss_list = []
#val_acc_list = []
#def loss(itr,tra_img,tra_lab,val_img,val_lab,model):
#
#    tra_pre = model.call(tra_img)
#    loss = softmax_cross_entropy_with_logits(labels = tra_lab, logits = tra_pre)
#    loss = tf.reduce_mean(loss)
#    tra_pre_arg = tf.argmax(tra_pre, 1).numpy()
#    tra_tar = np.argmax(tra_lab, 1)
#    tra_acc = np.sum(tra_pre_arg == tra_tar)/len(tra_tar)
#
#    val_pre = model.call(val_img)
#    val_loss = softmax_cross_entropy_with_logits(labels = val_lab, logits = val_pre)
#    val_loss = tf.reduce_mean(val_loss)
#    val_pred_arg = tf.argmax(val_pre, 1).numpy()
#    val_tar = np.argmax(val_lab, 1)
#    val_acc = np.sum(val_pred_arg == val_tar)/len(val_tar)
#
#    tra_loss_list.append(loss.numpy())
#    tra_acc_list.append(tra_acc)
#    val_loss_list.append(val_loss.numpy())
#    val_acc_list.append(val_acc)
#    #100イテレーション毎に誤差を出力
#    if (itr+1) % 500 == 0:
#        print("step {}:\tloss = {}\taccuracy = {}\tval_loss = {}\tval_accuracy = {}".format(itr+1, loss.numpy(), tra_acc, val_loss.numpy(), val_acc))
#    return loss

def loss(itr,tra_img,tra_lab,model):

    tra_pre = model.call(tra_img)
    loss = softmax_cross_entropy_with_logits(labels = tra_lab, logits = tra_pre)
    loss = tf.reduce_mean(loss)
    tra_pre_arg = tf.argmax(tra_pre, 1).numpy()
    tra_tar = np.argmax(tra_lab, 1)
    tra_acc = np.sum(tra_pre_arg == tra_tar)/len(tra_tar)
    #100イテレーション毎に誤差を出力
    if (itr+1) % (5000/500) == 0:
        print("step {}:\tloss = {}\taccuracy = {}".format(itr+1, loss.numpy(), tra_acc))
    return loss


if __name__ == '__main__':
    height =28
    width = 28
    padded_height = 40
    padded_width = 40
    batch_size = 500
    #itration = 1500
    epoch = 20
    FILE_WEIGHT = "../data/weight.hdf5"
    #FILE_MODEL = "../data/model.h5"
    FILE_STF_IMG = "../data/stf_img.jpg"
    FILE_HISTRY = "../data/history.csv"
    FILE_CON_MAT_CSV = "../data/confusion_matrix.csv"
    FILE_CON_MAT_JPG = "../data/confusion_matrix.jpg"
    DIR_TEST_IMG = "../data/test_img.jpg"
    DIR_TRAIN_IMG = "../data/train_img.jpg"
    DIR_TENSORBOARD = "../data/tb/"
    DIR_CHECK = "../data/training_checkpoints"
    #1 データ読み込み
    mnist = read_data_sets("/data/mnist", one_hot=True)
    tra_img = mnist.train.images.reshape((-1,28,28,1))
    tra_lab = mnist.train.labels

    val_img = mnist.validation.images.reshape((-1,28,28,1))
    val_lab = mnist.validation.labels

    tes_img = mnist.test.images.reshape((-1,28,28,1))
    tes_lab = mnist.test.labels

    #2 パディング
    padded_tra_img = np.zeros((tra_img.shape[0],padded_height, padded_width, 1))
    padded_val_img = np.zeros((val_img.shape[0],padded_height, padded_width, 1))
    padded_tes_img = np.zeros((tes_img.shape[0],padded_height, padded_width, 1))
    o = int((40-28)/2)
    padded_tra_img[:,o:o+height, o:o+width,:] = tra_img
    padded_val_img[:,o:o+height, o:o+width,:] = val_img
    padded_tes_img[:,o:o+height, o:o+width,:] = tes_img

    #3 データ拡張
    image_generator = ImageDataGenerator(
               rotation_range=30,#30
               width_shift_range=0.25,#0.25
               height_shift_range=0.25,#0.25
               shear_range=20,#20
               zoom_range=[0.8,1.2],#[0.8,1.2]
               horizontal_flip=False,
               vertical_flip=False)

    image_generator.fit(padded_tra_img, augment=False, seed=None)
    image_generate_flow = image_generator.flow(padded_tra_img, batch_size=tra_img.shape[0], shuffle=False)
    tra_img = image_generate_flow.next()

    image_generator.fit(padded_val_img, augment=False, seed=None)
    image_generate_flow = image_generator.flow(padded_val_img, batch_size=val_img.shape[0], shuffle=False)
    val_img = image_generate_flow.next()

    image_generator.fit(padded_tes_img, augment=False, seed=None)
    image_generate_flow = image_generator.flow(padded_tes_img, batch_size=tes_img.shape[0], shuffle=False)
    tes_img = image_generate_flow.next()

    print(tra_img.shape[0],val_img.shape[0],tes_img.shape[0])
    
    row_imgs =[]
    one_img = np.ones((42,42))
    for r in range(5):
        colum_imgs =[]

        for c in range(10):
            one_img[1:41,1:41] = tra_img[c+r*10,:,:,0]
            colum_imgs.append(one_img*255)
        row_imgs.append(colum_imgs)
    cv2.imwrite(DIR_TRAIN_IMG, concat_tile(row_imgs))
    print("Wrote train image")

    row_imgs =[]
    for r in range(5):
        colum_imgs =[]

        for c in range(10):
            one_img[1:41,1:41] = tes_img[c+r*10,:,:,0]
            colum_imgs.append(one_img*255)
        row_imgs.append(colum_imgs)
    cv2.imwrite(DIR_TEST_IMG, concat_tile(row_imgs))
    print("Wrote test image")
    
    #4 モデル生成
    model = my_stn()
    #model = my_cnn()

    #5 訓練
    #optimizer = AdamOptimizer(learning_rate=1e-3)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    #global_step = tf.train.get_or_create_global_step()
#
#    logdir = "./tb/"
#    writer = tf.contrib.summary.create_file_writer(DIR_TENSORBOARD)
#    writer.set_as_default()
#
#    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, global_step=tf.train.get_or_create_global_step())
#    manager = tf.contrib.checkpoint.CheckpointManager(checkpoint, directory=DIR_CHECK, max_to_keep=20)
#    status = checkpoint.restore(manager.latest_checkpoint)
#
##    checkpoint_prefix = os.path.join(DIR_CHECK, "ckpt")
##    status = checkpoint.restore(tf.train.latest_checkpoint(DIR_CHECK))
#
#    summary_writer = summary.create_file_writer(DIR_TENSORBOARD)
#    with summary_writer.as_default(), summary.always_record_summaries():
#        for itr in range(itration):
#
#            #バッチ毎に分ける
#            batch_img = tra_img[itr*batch_size:(itr+1)*batch_size]
#            batch_label = tra_lab[itr*batch_size:(itr+1)*batch_size]
#
#            #誤差の最小化
#            optimizer.minimize(lambda: loss(itr,batch_img,batch_label,model))
#        manager.save()

    del padded_tra_img,padded_val_img,padded_tes_img,image_generator

    tra_loss_list = []
    tra_acc_list = []
    val_loss_list = []
    val_acc_list = []
    train_num = tra_img.shape[0]
    tra_loss = 0
    tra_acc = 0
    #バッチ毎に分ける
    for batch in range(train_num//batch_size):
        batch_img = tra_img[batch*batch_size:(batch+1)*batch_size,:,:,:]
        batch_lab = tra_lab[batch*batch_size:(batch+1)*batch_size]
        
        batch_pre = model.call(batch_img)
        _tra_loss = softmax_cross_entropy_with_logits(labels = batch_lab, logits = batch_pre)
        tra_loss = tf.reduce_mean(_tra_loss).numpy() + tra_loss
        
        tra_pre_arg = tf.argmax(batch_pre, 1).numpy()
        tra_tar = np.argmax(batch_lab, 1)
        tra_acc = np.sum(tra_pre_arg == tra_tar)+ tra_acc

    tra_loss_list.append(tra_loss/(train_num//batch_size))
    tra_acc_list.append(tra_acc/train_num)
    
    val_pre = model.call(val_img[:,:,:,:])
    val_loss = softmax_cross_entropy_with_logits(labels = val_lab, logits = val_pre)
    val_loss = tf.reduce_mean(val_loss).numpy()
    val_pre_arg = tf.argmax(val_pre, 1).numpy()
    val_tar = np.argmax(val_lab, 1)
    val_acc = np.sum(val_pre_arg == val_tar)/len(val_tar)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    print("traloss:",tra_loss/(train_num//batch_size),"traacc",tra_acc/train_num,"val_loss",val_loss,"val_acc",val_acc)
    del val_pre,val_loss,val_pre_arg,val_tar,val_acc
    
    
    for e in range(epoch):
        print("Epoch:",e)
        tra_loss = 0
        tra_acc = 0
        
        
        #バッチ毎に分ける
        for batch in range(train_num//batch_size):
            batch_img = tra_img[batch*batch_size:(batch+1)*batch_size,:,:,:]
            batch_lab = tra_lab[batch*batch_size:(batch+1)*batch_size]

            #誤差の最小化
            #optimizer.minimize(lambda: loss(batch+e*(train_num//batch_size),batch_img,batch_label,val_img,val_lab,model))
            optimizer.minimize(lambda: loss(batch+e*(train_num//batch_size),batch_img,batch_lab,model))
            
            batch_pre = model.call(batch_img)
            _tra_loss = softmax_cross_entropy_with_logits(labels = batch_lab, logits = batch_pre)
            tra_loss = tf.reduce_mean(_tra_loss).numpy() + tra_loss
            

            tra_pre_arg = tf.argmax(batch_pre, 1).numpy()
            tra_tar = np.argmax(batch_lab, 1)
            tra_acc = np.sum(tra_pre_arg == tra_tar)+ tra_acc

        tra_loss_list.append(tra_loss/(train_num//batch_size))
        tra_acc_list.append(tra_acc/train_num)
        
        val_pre = model.call(val_img[:,:,:,:])
        val_loss = softmax_cross_entropy_with_logits(labels = val_lab, logits = val_pre)
        val_loss = tf.reduce_mean(val_loss).numpy()
        val_pre_arg = tf.argmax(val_pre, 1).numpy()
        val_tar = np.argmax(val_lab, 1)
        val_acc = np.sum(val_pre_arg == val_tar)/len(val_tar)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        del val_pre,val_loss,val_pre_arg,val_tar,val_acc
        
        

        
        
    row_imgs = []
    for r in range(5):
        colum_imgs =[]
        for c in range(10):
            one_img = np.ones((42,42))
            pre_img = model.stn_img(tes_img[c+r*10,:,:,:].reshape(1,40,40,1))
            one_img[1:41,1:41] = pre_img.numpy().reshape(40,40)
            one_img = one_img*255
            colum_imgs.append(one_img)
        row_imgs.append(colum_imgs)
    cv2.imwrite(FILE_STF_IMG, concat_tile(row_imgs))
    print("Wrote test image")
    del row_imgs
    
    evaluation = np.zeros((len(tra_loss_list),4))
    evaluation[:,0] = tra_loss_list[:]
    evaluation[:,1] = val_loss_list[:]
    evaluation[:,2] = tra_acc_list[:]
    evaluation[:,3] = val_acc_list[:]
    df = pd.DataFrame(evaluation, columns = ['loss','val_loss','accuracy','val_accuracy'])
    df.to_csv(FILE_HISTRY,index=True)
    #STNのテスト画像出力
    del df, evaluation


    tes_pre = model.call(tes_img[:,:,:,:])

    y_true = np.argmax(tes_pre, axis=1)
    y_pred = np.argmax(tes_lab, axis=1)
    print("TestAccuracy:",np.sum(y_pred == y_true)/len(y_true))
    cnf_matrix = confusion_matrix(y_true,y_pred)
    np.savetxt(FILE_CON_MAT_CSV, cnf_matrix, delimiter=",")
#    cnf_matrix = np.loadtxt(FILE_CON_MAT_CSV,delimiter=",")
    labels = ["0","1","2","3","4","5","6","7","8","9"]
    #print(classification_report(y_true, y_pred,target_names=labels))

    plt.figure(figsize = (10,7))
    #plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cmx_data.Blues)
    df_cmx = pd.DataFrame(cnf_matrix, index=labels, columns=labels)
    sns.heatmap(df_cmx, cmap="Blues", annot=True)
    plt.rcParams["font.size"] = 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.ylabel('Truth',fontsize=40)
    plt.xlabel('Prediction',fontsize=40)
    plt.tight_layout()
    plt.savefig(FILE_CON_MAT_JPG)