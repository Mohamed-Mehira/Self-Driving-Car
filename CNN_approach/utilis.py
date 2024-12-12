import pandas as pd
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from imgaug import augmenters as aug
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense, Lambda, Dropout
from keras.optimizers import Adam

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def getName(filePath):
    return filePath.split('\\')[-1]


def importDataInfo(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)
    print(data)
    data['Center'] = data['Center'].apply(getName)
    data.drop(['Left', 'Right', 'Throttle', 'Brake', 'Speed'], axis=1, inplace=True)
    # print(data)
    print('Total Images Imported', data.shape[0])
    return data


def balanceData(data, display=True):
    nBins = 31
    samplesPerBin = 2500
    hist, bins = np.histogram(data['Steering'], nBins)
    print(hist, bins)

    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        print(center)
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.show()


    # the number of 0 value steering angles is overwhelming, so we set a threshold
    zeroSteeringList = []
    for i in range(len(data)):
        if bins[15] <= data['Steering'][i] <= bins[16]:
            zeroSteeringList.append(i)
    zeroSteeringList = shuffle(zeroSteeringList)
    zeroSteeringList = zeroSteeringList[samplesPerBin:]

    print('Removed Images:', len(zeroSteeringList))
    data.drop(data.index[zeroSteeringList], inplace=True)
    print('Remaining Images:', len(data))

    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.show()


def loadData(path, data):
    imgsList = []
    steeringList = []

    for i in range(len(data)):
        indexed_data = data.iloc[i]
        img = mpimg.imread(f'{path}/IMG/{indexed_data[0]}')
        # img = preProcess(img)    # in case of no augmentation
        imgsList.append(img)
        steeringList.append(float(indexed_data[1]))

    imgsList = np.asarray(imgsList)
    steeringList = np.asarray(steeringList)
    return imgsList, steeringList


def preProcess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


def augmentImage(img, steering):
    if np.random.rand() < 0.5:
        pan = aug.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = aug.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = aug.Multiply((0.2, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering


def batchGen(imgsList, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imgsList) - 1)

            if trainFlag:
                img, steering = augmentImage(imgsList[index], steeringList[index])
            else:
                img = imgsList[index]
                steering = steeringList[index]

            img = preProcess(img)
            imgBatch.append(img)
            steeringBatch.append(steering)

        yield np.asarray(imgBatch), np.asarray(steeringBatch)


def createModel():
    model = Sequential()

    # model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    # model.add(Convolution2D(24, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(learning_rate=0.0001), loss='mse')
    return model


