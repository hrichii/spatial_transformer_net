import cv2
import numpy as np
from numpy.random import *
#from IPython.display import display, Image


#def display_cv_image(image, format='.png'):
#    decoded_bytes = cv2.imencode(format, image)[1].tobytes()
#    display(Image(data=decoded_bytes))


def add_edge(img):
    H, W = img.shape
    WID = int(np.max(img.shape) * 2**0.5)
    e_img = np.zeros((WID, WID))
    e_img[int((WID-H)/2):int((WID+H)/2),
          int((WID-W)/2):int((WID+W)/2)] = img
    return e_img


def translation_matrix(tx, ty):
    return np.array([[1, 0, -tx],
                     [0, 1, -ty],
                     [0, 0, 1]])


def rotation_matrix(a):
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a),  np.cos(a), 0],
                     [        0,          0, 1]])


def shear_matrix(mx, my):
    return np.array([[1, -mx, 0],
                     [-my, 1, 0],
                     [0,  0, 1]])     


def scaling_matrix(sx, sy):
    return np.array([[1/sx, 0, 0],
                     [0, 1/sy, 0],
                     [0,  0, 1]])


#def affin(img, m):
if __name__ == '__main__':
    img = cv2.imread("FILENAME.png")
    m = rotation_matrix(359.9*rand())                                           #回転
    m = np.dot(scaling_matrix((1.5-0.8)*rand()+0.8, (1.5-0.8)*rand()+0.8),m)    #拡縮
    m = np.dot(shear_matrix(0.1*rand(), 0.6*rand()),m)                          #シーア変換
    m = np.dot(translation_matrix(0.1*rand(),0.1*rand()),m)                     #平行移動
    WID = np.max(img.shape)
    x = np.tile(np.linspace(-1, 1, WID).reshape(1, -1), (WID, 1))
    y = np.tile(np.linspace(-1, 1, WID).reshape(-1, 1), (1, WID))
    p = np.array([[x, y, np.ones(x.shape)]])
    dx, dy, _ = np.sum(p * m.reshape(*m.shape, 1, 1), axis=1)
    u = np.clip((dx + 1) * WID / 2, 0, WID-1).astype('i')
    v = np.clip((dy + 1) * WID / 2, 0, WID-1).astype('i')
    newimg=img[v, u]
    cv2.imwrite("FILENAME_edited.png",newimg)