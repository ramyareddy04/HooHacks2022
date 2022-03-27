import os
from turtle import pd

from keras.preprocessing import image
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical
import random, shutil
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
def generator(dir, gen=image.ImageDataGenerator(rescale=1. / 255), shuffle=True, batch_size=1, target_size=(24, 24),
              class_mode='categorical'):  #classmode can be binary or categorical
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale',
                                   class_mode=class_mode, target_size=target_size)

batchSize = 32
targetSize = (24, 24)
train_batch = generator('data/train', shuffle=True, batch_size=batchSize, target_size=targetSize)
valid_batch = generator('data/train', shuffle=True, batch_size=batchSize, target_size=targetSize)
SPE = len(train_batch.classes) // batchSize
VS = len(valid_batch.classes) // batchSize
print(SPE, VS)

img,labels= next(train_batch)
print("This is img shape")
print(img.shape)
# print(labels.size)
model = Sequential([

    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    # 32 convolution filters used each of size 3x3
    # again
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),

    # 64 convolution filters used each of size 3x3
    # choose the best features via pooling

    # randomly turn neurons on and off to improve convergence
    Dropout(0.25),
    # flatten since too many dimensions, we only want a classification output
    Flatten(),
    # fully connected to get all relevant data
    Dense(32, activation='relu'),
    # one more dropout for convergence' sake :)
    Dropout(0.25),
    # output a softmax to squash the matrix into output probabilities
    Dense(4, activation='softmax') #use softmax when doing multiclass/ use sigmoid when doing binary
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_batch, validation_data=valid_batch, epochs=20, steps_per_epoch=SPE, validation_steps=VS)

model.save('models/test10.h5', overwrite=True)
