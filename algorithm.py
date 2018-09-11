"""

Algorithm tester code
includes SVM, RandomForest, Ensemble Methods and KNN
Includes few Deep learning models as well

"""

import h5py
import numpy as np
import os
from matplotlib import pyplot
from keras.models import  Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense

hf = h5py.File('dat.hdf5', 'r')
X = hf.get('Images_2D')
y = hf.get('Labels')
X = np.array(X)
y = np.array(y)
hf.close()
X = X[0:235, 16:192, 0:176]


def store_in_dir(X):
    if not os.path.isdir("data_x"):
        os.makedirs("data_x")
    os.chdir("data_x")
    r = 234
    for i in range(r):
        pyplot.imsave("%d_img.png"%i,X[i])
        print("doing")


def make_dataset(x,y):
    xx = np.asarray(x).reshape(235, 176, 176, 1)
    # yy = np.asarray(y).reshape(1,235)
    return xx, y


x, y = make_dataset(X,y)

from keras.utils import to_categorical
y = to_categorical(y)


def model(input_shape = (176, 176, 1)):
    layers = Sequential()
    layers.add(Conv2D(5, (40, 40), padding='same', activation='relu', input_shape=input_shape))
    layers.add(Conv2D(3, (20, 20)))
    layers.add(Conv2D(3, (15, 15)))
    layers.add(MaxPooling2D(pool_size=(2, 2)))
    layers.add(Dropout(0.25))
    layers.add(Flatten())
    layers.add(Dense(20))
    layers.add(Dense(2, activation='softmax'))
    return layers


model = model()

ep = 10
model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x[:200], y[:200], epochs=ep, verbose=1, validation_data=(x[200:230], y[200:230]))

model.evaluate(x, y)
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)


# Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['acc'], 'r', linewidth=3.0)
plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
plt.show()