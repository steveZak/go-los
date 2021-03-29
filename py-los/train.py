import json
import os
import cv2
import random
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, models, optimizers
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
from ctypes import *

pigo = cdll.LoadLibrary('./talkdet.so')

MAX_NDETS = 2024
ARRAY_DIM = 6

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# define class GoPixelSlice to map to:
# C type struct { void *data; GoInt len; GoInt cap; }
class GoPixelSlice(Structure):
    _fields_ = [
        ("pixels", POINTER(c_ubyte)), ("len", c_longlong), ("cap", c_longlong),
    ]

# Obtain the camera pixels and transfer them to Go through Ctypes
def processFrame(pixs):
    dets = np.zeros(ARRAY_DIM * MAX_NDETS, dtype=np.float32)
    pixels = cast((c_ubyte * len(pixs))(*pixs), POINTER(c_ubyte))

    # call FindFaces
    faces = GoPixelSlice(pixels, len(pixs), len(pixs))
    pigo.FindFaces.argtypes = [GoPixelSlice]
    pigo.FindFaces.restype = c_void_p

    # Call the exported FindFaces function from Go.
    ndets = pigo.FindFaces(faces)
    data_pointer = cast(ndets, POINTER((c_longlong * ARRAY_DIM) * MAX_NDETS))

    if data_pointer:
        buffarr = ((c_longlong * ARRAY_DIM) *
                   MAX_NDETS).from_address(addressof(data_pointer.contents))
        res = np.ndarray(buffer=buffarr, dtype=c_longlong,
                         shape=(MAX_NDETS, ARRAY_DIM,))

        dets_len = res[0][0]
        res = np.delete(res, 0, 0)
        dets = list(res.reshape(-1, ARRAY_DIM))[0:dets_len*19]
        return dets

# class CNN():
#     def __init__():
        

def createNN(bs=50):
    model = models.Sequential()
    model.add(layers.Conv2D(bs, (3, 3), activation='relu', input_shape=(100, 200, 3)))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(bs, (3, 3), activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1000))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1000))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(700))
    # model.add(layers.Dense(300))
    model.add(layers.Dense(40, activation="softmax"))
    print(model.summary())
    return model

def forward(model, X):
    return model(X)

def backprop(model, X, Y):
    opt = optimizers.Adam()
    with tf.GradientTape() as tape:
        predictions = model(X, training=True)
        _loss = losses.kullback_leibler_divergence(Y, predictions)
        # _loss = losses.sparse_categorical_crossentropy([np.argmax(y) for y in Y], predictions)
    gradients = tape.gradient(_loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return _loss

def testFrame(model, img):
    model = tf.saved_model.load(model)
    image = cv2.resize(img[100:2150, :, :], (640, 480))
    pixs = np.ascontiguousarray(image[:,:, 1].reshape((image.shape[0], image.shape[1])))
    pixs1 = pixs.flatten()
    dets = processFrame(pixs1)
    if len(dets)> 0:
        mouth = cv2.resize(image[dets[16][0]-5:dets[15][0]+5, dets[14][1]-5:dets[17][1]+5], (200, 100))
        # cv2.imwrite('cool2.png', mouth)
        y = forward(model, mouth[np.newaxis, ...].astype(np.float32))
        return y
    return None

labels = ['ah', 'b', 'aw', 't', 'hh', 'ae', 'f', 'w', 'ey', 'ih', 'iy', 'n', 'eh', 's', 'g', 'd', 'uw', 'y', 'ao', 'r', 'k', 'dh', 'm', 'ow', 'er', 'l', 'jh', 'oy', 'z', 'ay', 'v', 'sh', 'ng', 'aa', 'ch', 'th', 'p', 'zh', 'uh', 'sil']

# train
# large model dense architecture: 1000 / 1000 / 700
# small model dense architecture: 1000 / 1000 / 300
# x = np.load("X.npy", allow_pickle=True)
# y = np.load("Y.npy", allow_pickle=True)
# x = np.concatenate(x, axis=0)
# y = np.concatenate(y, axis=0)
# rng_state = np.random.get_state()
# np.random.shuffle(x)
# np.random.set_state(rng_state)
# np.random.shuffle(y)
# model = createNN()
# _losses = []
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# for e in range(60):
#     out = forward(model, x[5][np.newaxis, ...].astype(np.float32)/255.0)
#     print(out)
#     print(labels[np.argmax(out)])
#     train_loss.reset_states()
#     for i in range(int(len(x)/50)):
#         _loss = backprop(model, x[i:i+50].astype(np.float32)/255.0, y[i:i+50].astype(np.float32))
#         train_loss(_loss)
#     print(train_loss.result())
#     _losses.append(train_loss.result())
#     # print(_loss)
#     print(e)
# model.save("model_large")

# plt.plot([i for i in range(e+1)], _losses, 'ro')
# plt.show()
