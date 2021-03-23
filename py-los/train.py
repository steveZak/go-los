import json
import os
import cv2
import imageio
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, losses, models, optimizers
from ctypes import *

pigo = cdll.LoadLibrary('./talkdet.so')

MAX_NDETS = 2024
ARRAY_DIM = 6

MOUTH_AR_THRESH = 0.2
MOUTH_AR_CONSEC_FRAMES = 5

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

        # The first value of the buffer aray represents the buffer length.
        dets_len = res[0][0]
        # print(dets_len)
        res = np.delete(res, 0, 0)  # delete the first element from the array

        # We have to multiply the detection length with the total
        # detection points(face, pupils and facial lendmark points), in total 18
        dets = list(res.reshape(-1, ARRAY_DIM))[0:dets_len*19]
        return dets

def getImages(reader, start, end, frame_rate=30):
    frames = []
    print(round(30*start))
    print(round(30*end))
    for i in range(round(30*start), round(30*end)):
        image = reader.get_data(i)
        # image = reader[i]
        # break
        b,g,r = cv2.split(image)           # get b, g, r
        image = cv2.resize(cv2.merge([r,g,b])[190:1100, 0:1400, :], (640, 480))
        pixs = np.ascontiguousarray(image[:,:, 1].reshape((image.shape[0], image.shape[1])))
        pixs1 = pixs.flatten()
        dets = process_frame(pixs1)
        if len(dets)> 0:
            mouth = cv2.resize(image[dets[16][0]-5:dets[15][0]+5, dets[14][1]-5:dets[17][1]+5], (200, 100))
            # cv2.imwrite('cool1.png', mouth)
            frames.append(mouth)
    return frames

def getReader():
    filename = 'gatsby2.MOV'
    reader = imageio.get_reader(filename,  'ffmpeg')
    return reader

def oneHot(idx, length):
    onehot = np.zeros((1, length))
    onehot[0, idx] = 1
    return onehot

def getDataset(reader, phonemes):
    X, Y = [], []
    idx = 0
    # print(len(phonemes[phoneme]))
    for phoneme in phonemes:
        # print(phoneme)
        images = np.array([], dtype=np.int64).reshape(0, 100, 200, 3)
        labels = np.array([], dtype=np.int64).reshape(0, len(phonemes.keys()))
        i=0
        for frames in phonemes[phoneme]:
            i+=1
            _images = getImages(reader, frames[0], frames[1])
            for image in _images:
                images = np.vstack([images, image[np.newaxis, ...]])
                labels = np.vstack([labels, oneHot(idx, len(phonemes))])
        X.append(labels)
        Y.append(images)
        print(len(X))
        print(len(Y))
        idx+=1
        if idx>1:
            break
    return X, Y

# class CNN():
#     def __init__():
        

def createNN():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(640, 480, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(640, 480, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(640, 480, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(300))
    model.add(layers.Dense(39))
    return model

def forward(model, X):
    return model(X)

def backprop(model, X, Y):
    Y_hat = forward(model, X)
    loss = losses.BinaryCrossentropy()
    opt = optimizers.SGD()
    opt.minimize(loss)
    #backprop the model


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
    phonemes.pop("sil", None)
    # combine similar phonemes
    # phonemes[].extend(phonemes[])
    # phonemes.pop()
    return phonemes


# phonemes = readPhonemes("gatsby2resfinal.json")
# reader = getReader()
# X, Y = getDataset(reader, phonemes)
# with open('X.npy', 'wb') as f:
#     np.save(f, X)
# with open('Y.npy', 'wb') as f:
#     np.save(f, Y)
# print(len(Y))
# print(Y[0])
# x = np.load("Y.npy")
# print(x)
createNN()