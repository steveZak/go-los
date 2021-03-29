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