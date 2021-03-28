import json
import os
import cv2
import random
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, models, optimizers
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
def process_frame(pixs):
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

def getImages(frame_bounds, frame_rate=30):
    frames = []
    filename = 'gatsby2.MOV'
    cap = cv2.VideoCapture(filename)
    j = 0
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        # print(j)
        res, image = cap.read()
        if i >= math.ceil(cap.get(5)*frame_bounds[j][0])+1 and i <= math.floor(cap.get(5)*frame_bounds[j][1])-1:
            image = cv2.resize(image[190:1100, 0:1400, :], (640, 480))
            pixs = np.ascontiguousarray(image[:,:, 1].reshape((image.shape[0], image.shape[1])))
            pixs1 = pixs.flatten()
            dets = process_frame(pixs1)
            if len(dets)> 0:
                mouth = cv2.resize(image[dets[16][0]-5:dets[15][0]+5, dets[14][1]-5:dets[17][1]+5], (200, 100))
                # cv2.imwrite('cool2.png', mouth)
                frames.append(mouth)
            print("time = "+str(i/cap.get(5)))
        if i >= math.floor(cap.get(5)*frame_bounds[j][1]):
            j+=1
            if j==len(frame_bounds):
                return frames
    return frames

def getReader():
    filename = 'gatsby2.MOV'
    reader = cv2.VideoCapture(filename)
    return reader

def oneHot(idx, length):
    onehot = np.zeros((1, length))
    onehot[0, idx] = 1
    return onehot

def getDataset(phonemes):
    X, Y = [], []
    idx = 0
    for phoneme in phonemes:
        print(phoneme)
        images = np.array([], dtype=np.float32).reshape(0, 100, 200, 3)
        labels = np.array([], dtype=np.float32).reshape(0, len(phonemes.keys()))
        _images = getImages(phonemes[phoneme])
        for image in _images:
            images = np.vstack([images, image[np.newaxis, ...]])
            labels = np.vstack([labels, oneHot(idx, len(phonemes))])
        X.append(images)
        Y.append(labels)
        idx+=1
    return X, Y

# class CNN():
#     def __init__():
        

def createNN(bs=50):
    model = models.Sequential()
    model.add(layers.Conv2D(bs, (3, 3), activation='relu', input_shape=(100, 200, 3)))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(bs, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1000))
    model.add(layers.Dense(1000))
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


def readPhonemes(fname):
    phonemes = {}
    with open(fname, 'r') as file:
        data = json.load(file)
    for word in data["words"]:
        if word["case"] == "success":
            phoneme_start = 0 # phoneme start time
            for phone in word["phones"]:
                if phone["phone"] in phonemes:
                    phonemes[phone["phone"]].append([phoneme_start + word["start"], phoneme_start + word["start"] + phone["duration"]])
                else:
                    phonemes[phone["phone"]] = [[phoneme_start + word["start"], phoneme_start + word["start"] + phone["duration"]]]
                phoneme_start += phone["duration"]
    # remove out of vocabulary and silent phonemes
    phonemes.pop("oov", None)
    # phonemes.pop("sil", None)
    # combine similar phonemes
    # phonemes[].extend(phonemes[])
    # phonemes.pop()
    return phonemes


def testFrame(model, img):
    model = tf.saved_model.load(model)
    image = cv2.resize(img[100:2150, :, :], (640, 480))
    pixs = np.ascontiguousarray(image[:,:, 1].reshape((image.shape[0], image.shape[1])))
    pixs1 = pixs.flatten()
    dets = process_frame(pixs1)
    if len(dets)> 0:
        mouth = cv2.resize(image[dets[16][0]-5:dets[15][0]+5, dets[14][1]-5:dets[17][1]+5], (200, 100))
        # cv2.imwrite('cool2.png', mouth)
        y = forward(model, mouth[np.newaxis, ...].astype(np.float32))
        return y
    return None


def predictFromVideo(model_name, filename):
    word = []
    model = tf.saved_model.load(model_name)
    cap = cv2.VideoCapture(filename)
    j = 0
    prev = None
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        res, image = cap.read()
        image = cv2.flip(cv2.resize(image[190:800, 0:1400, :], (640, 480)), 1)
        cv2.imwrite('resized.png', image)
        pixs = np.ascontiguousarray(image[:,:, 1].reshape((image.shape[0], image.shape[1])))
        pixs1 = pixs.flatten()
        dets = process_frame(pixs1)
        print(len(dets))
        if len(dets) > 0:
            mouth = cv2.resize(image[dets[16][0]-5:dets[15][0]+5, dets[14][1]-5:dets[17][1]+5], (200, 100))
            y = forward(model, mouth[np.newaxis, ...].astype(np.float32))
            print(y)
            res = labels[np.argmax(y)]
            print(res)
            cv2.imwrite('frame_predictions/' + res + str(i) + '.png', mouth)
            if prev==None or (prev == res and word[-1] is not res):
                word.append(res)
            prev = res
    return word


# build dataset
# phonemes = readPhonemes("gatsby2result.json")
# # print(phonemes.keys())
labels = ['ah', 'b', 'aw', 't', 'hh', 'ae', 'f', 'w', 'ey', 'ih', 'iy', 'n', 'eh', 's', 'g', 'd', 'uw', 'y', 'ao', 'r', 'k', 'dh', 'm', 'ow', 'er', 'l', 'jh', 'oy', 'z', 'ay', 'v', 'sh', 'ng', 'aa', 'ch', 'th', 'p', 'zh', 'uh', 'sil']
# # reader = getReader()
# X, Y = getDataset(phonemes)
# with open('X.npy', 'wb') as f:
#     np.save(f, X, allow_pickle=True)
# with open('Y.npy', 'wb') as f:
#     np.save(f, Y, allow_pickle=True)

# train
# large model structure: 1000 / 1000 / 700
# small model structure: 1000 / 1000 / 300
# x = np.load("X.npy", allow_pickle=True)
# y = np.load("Y.npy", allow_pickle=True)
# x = np.concatenate(x, axis=0)
# y = np.concatenate(y, axis=0)
# rng_state = np.random.get_state()
# np.random.shuffle(x)
# np.random.set_state(rng_state)
# np.random.shuffle(y)
# model = createNN()
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# print(train_loss.result())
# for e in range(10):
#     train_loss.reset_states()
#     for i in range(int(len(x)/50)):
#         _loss = backprop(model, x[i:i+50].astype(np.float32)/255.0, y[i:i+50].astype(np.float32))
#         # print(float(i)/len(indices))
#         train_loss(_loss)
#     print(train_loss.result())
#     # print(_loss)
#     print(e)
# model.save("model_large")


# predict
# img = cv2.imread("m.jpg")
# out = testFrame("model_large", img)
# print(out)
# print(labels[np.argmax(out)])

# img = cv2.imread("ah.jpg")
# out = testFrame("model_large", img)
# print(out)
# print(labels[np.argmax(out)])


#predict video
word = predictFromVideo('model_large', 'example.MOV')
print(word)
