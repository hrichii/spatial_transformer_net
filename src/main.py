#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial transformer net
"""
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data as tfload
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

tfe.enable_eager_execution()

INPATH = "../data/"
filepath=INPATH+"FILENAME.png"
H = 28
W = 28

def write_gray2png(filepath, gray):
    """
    グレースケールの配列をpngに変換し，保存
    """
    H = gray.shape[0]
    W = gray.shape[1]
    binary = tf.image.encode_png(gray.reshape(H,W,1))
    tf.write_file(filepath, binary)


def pad_distort_im_fn(x):
    """ 
    画像を40x40にゼロパディングし，歪める

    """
    #ゼロパディング
    z = np.zeros((40, 40, 1))
    o = int((40-28)/2)
    z[o:o+28, o:o+28] = x
    x = z
    
    #画像をゆがめる
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    x = zoom(x)
    return x


def plot_images(dataset, n_images, samples_per_image):
    output = np.zeros((32 * n_images, 32 * samples_per_image, 3))

    row = 0
    for images in dataset.repeat(samples_per_image).batch(n_images):
        output[:, row*32:(row+1)*32] = np.vstack(images.numpy())
        row += 1

    plt.figure()
    plt.imshow(output)
    plt.show()
    
def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x

def color(x: tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

def zoom(x: tf.Tensor) -> tf.Tensor:
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))


#1 load data images(10000x28x28)
(x_train, y_train), (x_test, y_test) = tfload()
x = x_train[0,:,:].reshape(H,W,1)
x = pad_distort_im_fn(x)
x = tf.cast(x,tf.uint8).numpy()
output_image = write_gray2png(filepath,x)
#
#degrees = 120
#a = tf.contrib.image.rotate(x_train[0,:,:], degrees * math.pi / 180, interpolation='BILINEAR').numpy()
#a = tf.contrib.image.rotate(x_train[0,:,:], degrees * math.pi / 180, interpolation='BILINEAR').numpy()
#
#output_image = write_gray2png(filepath,a)