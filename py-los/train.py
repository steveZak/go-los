import json
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, models, optimizers
from ctypes import *

pigo = cdll.LoadLibrary('./talkdet.so')

MAX_NDETS = 2024
ARRAY_DIM = 6

MOUTH_AR_THRESH = 0.2
MOUTH_AR_CONSEC_FRAMES = 5

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

        # The first value of the buffer aray represents the buffer length.
        dets_len = res[0][0]
        # print(dets_len)
        res = np.delete(res, 0, 0)  # delete the first element from the array

        # We have to multiply the detection length with the total
        # detection points(face, pupils and facial lendmark points), in total 18
        dets = list(res.reshape(-1, ARRAY_DIM))[0:dets_len*19]
        return dets

def getImages(frame_bounds, frame_rate=30):
    frames = []
    filename = 'gatsby2.MOV'
    # reader = imageio.get_reader(filename,  'ffmpeg')
    cap = cv2.VideoCapture(filename)
    j = 0
    print(frame_bounds)
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        # print(j)
        res, image = cap.read()
        if i >= round(cv2.CAP_PROP_FPS*frame_bounds[j][0]) and i <= round(cv2.CAP_PROP_FPS*frame_bounds[j][1]):
            # image = reader.get_data(i)
            # res, image = cap.retrieve(flag=7)
            # image = reader[i]
            # break
            # b,g,r = cv2.split(image)           # get b, g, r
            # image = cv2.resize(cv2.merge([r,g,b])[190:1100, 0:1400, :], (640, 480))
            image = cv2.resize(image[190:1100, 0:1400, :], (640, 480))
            pixs = np.ascontiguousarray(image[:,:, 1].reshape((image.shape[0], image.shape[1])))
            pixs1 = pixs.flatten()
            dets = process_frame(pixs1)
            if len(dets)> 0:
                mouth = cv2.resize(image[dets[16][0]-5:dets[15][0]+5, dets[14][1]-5:dets[17][1]+5], (200, 100))
                cv2.imwrite('cool2.png', mouth)
                frames.append(mouth)
            print("time="+str(i/cv2.CAP_PROP_FPS))
        if i >= round(cv2.CAP_PROP_FPS*frame_bounds[j][1]):
            j+=1
            if j==len(frame_bounds):
                return frames
        if i==6400:
            return frames
            # print("time="+str(frame_bounds[j][1]))
    return frames

def getReader():
    filename = 'gatsby2.MOV'
    # reader = imageio.get_reader(filename,  'ffmpeg')
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
        print(len(phonemes[phoneme]))
        # print(phoneme)
        images = np.array([], dtype=np.float32).reshape(0, 100, 200, 3)
        labels = np.array([], dtype=np.float32).reshape(0, len(phonemes.keys()))
        _images = getImages(phonemes[phoneme])
        for image in _images:
            images = np.vstack([images, image[np.newaxis, ...]])
            labels = np.vstack([labels, oneHot(idx, len(phonemes))])
        X.append(images)
        Y.append(labels)
        idx+=1
        # if idx>1:
        #     break
    return X, Y

# class CNN():
#     def __init__():
        

def createNN(bs=50):
    model = models.Sequential()
    model.add(layers.Conv2D(bs, (3, 3), activation='relu', input_shape=(100, 200, 3)))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(bs, (3, 3), activation='relu', input_shape=(100, 200, 3)))
    model.add(layers.MaxPooling2D((3, 3)))
    # model.add(layers.Conv2D(bs, (3, 3), activation='relu', input_shape=(100, 200, 3)))
    # model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Flatten())
    # model.add(layers.Dense(1000))
    # model.add(layers.Dense(1000))
    model.add(layers.Dense(500))
    model.add(layers.Dense(40, activation="softmax"))
    print(model.summary())
    return model

def forward(model, X):
    return model(X)

def backprop(model, X, Y):
    # model.trainable_variables
    # Y_hat = forward(model, X)
    # loss = losses.MSE()
    # opt = optimizers.Adam(learning_rate=0.005)
    opt = optimizers.Adam()
    # opt.minimize(loss)
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(X, training=True)
        # loss = loss_object(Y, predictions)
        # _loss = loss(Y, predictions)
        _loss = losses.kullback_leibler_divergence(Y, predictions)
    gradients = tape.gradient(_loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return _loss
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
    # phonemes.pop("sil", None)
    # combine similar phonemes
    # phonemes[].extend(phonemes[])
    # phonemes.pop()
    return phonemes

def getShuffledIndices(y):
    indices = []
    for i in range(len(y)):
        for j in range(len(y[i])):
            indices.append([i, j])
    random.shuffle(indices)
    return indices

def testFrame(img):
    model = tf.saved_model.load("model")
    image = cv2.resize(img[100:2150, :, :], (640, 480))
    pixs = np.ascontiguousarray(image[:,:, 1].reshape((image.shape[0], image.shape[1])))
    pixs1 = pixs.flatten()
    dets = process_frame(pixs1)
    if len(dets)> 0:
        mouth = cv2.resize(image[dets[16][0]-5:dets[15][0]+5, dets[14][1]-5:dets[17][1]+5], (200, 100))
        cv2.imwrite('cool2.png', mouth)
        y = forward(model, mouth[np.newaxis, ...].astype(np.float32))
        return y
    return None

# build dataset
# phonemes = readPhonemes("gatsby2resfinal.json")
# print(phonemes.keys())
labels = ['ah', 'b', 'aw', 't', 'hh', 'ae', 'f', 'w', 'ey', 'ih', 'iy', 'n', 'eh', 's', 'g', 'd', 'uw', 'y', 'ao', 'r', 'k', 'dh', 'm', 'ow', 'er', 'l', 'jh', 'oy', 'z', 'ay', 'v', 'sh', 'ng', 'aa', 'ch', 'th', 'p', 'zh', 'uh', 'sil']
# # reader = getReader()
# X, Y = getDataset(phonemes)
# with open('X.npy', 'wb') as f:
#     np.save(f, X, allow_pickle=True)
# with open('Y.npy', 'wb') as f:
#     np.save(f, Y, allow_pickle=True)

# train
x = np.load("X.npy", allow_pickle=True)
y = np.load("Y.npy", allow_pickle=True)
# x = np.array([], dtype=np.float32).reshape(0, 100, 200, 3)
# for i in range(len(_x)):
#     x = np.stack(_x[i], axis=0)
#     y = np.stack(_y[i], axis=0)
x = np.concatenate(x, axis=0)
y = np.concatenate(y, axis=0)
rng_state = np.random.get_state()
np.random.shuffle(x)
np.random.set_state(rng_state)
np.random.shuffle(y)
model = createNN()
# indices = getShuffledIndices(y)
# print(indices[0:5])
train_loss = tf.keras.metrics.Mean(name='train_loss')
print(train_loss.result())
print(x.shape)
for e in range(10):
    img = cv2.imread("m.jpg")
    out = testFrame(img)
    print(out)
    print(labels[np.argmax(out)])

    img = cv2.imread("ah.jpg")
    out = testFrame(img)
    print(out)
    print(labels[np.argmax(out)])
    # train_loss.reset_states()
    for i in range(int(len(x)/50)):
        # _loss = backprop(model, x[indices[i:i+50][0]][indices[i:i+50][1]].astype(np.float32), y[indices[i:i+50][0]][indices[i:i+50][1]].astype(np.float32))
        _loss = backprop(model, x[i:i+50].astype(np.float32)/255.0, y[i:i+50].astype(np.float32))
        # print(float(i)/len(indices))
    train_loss(_loss)
    print(train_loss.result())
    # print(_loss)
    print(e)
# out = forward(model, x[0][0:5].astype(np.float32))
model.save("model3")

# predict
img = cv2.imread("m.jpg")
out = testFrame(img)
print(out)
print(labels[np.argmax(out)])

img = cv2.imread("ah.jpg")
out = testFrame(img)
print(out)
print(labels[np.argmax(out)])