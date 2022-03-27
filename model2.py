import os
import pandas as pd

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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
train_data = pd.read_csv('trainBinary.csv', dtype=str)
batchSize = 32
targetSize = (24, 24)

datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.1)
train_batch = datagen.flow_from_dataframe(dataframe=train_data, directory="data/finData/", x_col="filename", y_col="label",
                                          subset="training",
                                          batch_size=batchSize, seed=42,
                                          shuffle=True,
                                          class_mode="binary",
                                          target_size=targetSize)
valid_batch = datagen.flow_from_dataframe(dataframe=train_data, directory="data/finData/", x_col="filename", y_col="label",
                                          subset="validation",
                                          batch_size=batchSize, seed=42,
                                          shuffle=True,
                                          class_mode="binary",
                                          target_size=targetSize)
# train_batch = generator('data/train', shuffle=True, batch_size=batchSize, target_size=targetSize)
#valid_batch = generator('data/train', shuffle=True, batch_size=batchSize, target_size=targetSize)

STEP_SIZE_TRAIN=train_batch.n//train_batch.batch_size
STEP_SIZE_VALID=valid_batch.n//valid_batch.batch_size
print(STEP_SIZE_TRAIN, STEP_SIZE_VALID)

img, labels = next(train_batch)
print("This is img shape")
print(img.shape)
# print(labels.size)
model = Sequential([

    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 3)),
    MaxPooling2D(pool_size=(1, 1)),
    Dropout(0.35),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    # 32 convolution filters used each of size 3x3
    # # again
    Dropout(0.40),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Dropout(0.35),
    # 64 convolution filters used each of size 3x3
    # choose the best features via pooling
    Dense(32, activation='relu'),
    # randomly turn neurons on and off to improve convergence
    Dropout(0.25),
    # flatten since too many dimensions, we only want a classification output
    Flatten(),
    # fully connected to get all relevant data
    Dense(16, activation='relu'),
    # one more dropout for convergence' sake :)
    Dropout(0.25),
    # output a softmax to squash the matrix into output probabilities
    Dense(1, activation='sigmoid')  # use softmax when doing multiclass/ use sigmoid when doing binary
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_batch, validation_data=valid_batch, epochs=50, steps_per_epoch=STEP_SIZE_TRAIN, validation_steps=STEP_SIZE_VALID)

model.save('models/categoricalTest7.h5', overwrite=True)
