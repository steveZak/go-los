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
from train import forward, processFrame

pigo = cdll.LoadLibrary('./talkdet.so')

MAX_NDETS = 2024
ARRAY_DIM = 6

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


labels = ['ah', 'b', 'aw', 't', 'hh', 'ae', 'f', 'w', 'ey', 'ih', 'iy', 'n', 'eh', 's', 'g', 'd', 'uw', 'y', 'ao', 'r', 'k', 'dh', 'm', 'ow', 'er', 'l', 'jh', 'oy', 'z', 'ay', 'v', 'sh', 'ng', 'aa', 'ch', 'th', 'p', 'zh', 'uh', 'sil']

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
        dets = processFrame(pixs1)
        print(len(dets))
        if len(dets) > 0:
            mouth = cv2.resize(image[dets[16][0]-5:dets[15][0]+5, dets[14][1]-5:dets[17][1]+5], (200, 100))
            y = forward(model, mouth[np.newaxis, ...].astype(np.float32)/255.0)
            print(y)
            res = labels[np.argmax(y)]
            print(res)
            cv2.imwrite('frame_predictions/' + res + str(i) + '.png', mouth)
            if prev==None or (prev == res and word[-1] is not res):
                word.append(res)
            prev = res
    return word


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
word = predictFromVideo('model_large', 'velvet.MOV')
print(word)
