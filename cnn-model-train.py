from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import load_model
from keras.utils import np_utils
from scipy.io import loadmat

import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback


print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

# workaround for no available gpus exception
def _get_available_gpus():
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus



batch_size = 128
num_epochs = 20
kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 1024

# num_train = 60000
# num_test = 10000

height, width, depth = 32, 32, 1
num_classes = 10

def load_data(path):
    data = loadmat(path)
    return data['X'], data['y']

def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)

X_train, y_train = load_data("train_32x32.mat")
X_test, y_test = load_data("test_32x32.mat")

# num_classes = np.unique(y_train).shape[0]


# Reshape the image arrays
X_train, y_train = np.rollaxis(X_train, 3), y_train[:,0]
X_test, y_test = np.rollaxis(X_test, 3), y_test[:,0]

# print("test shape",X_test.shape)

X_train_gs = rgb2gray(X_train)
X_test_gs = rgb2gray(X_test)

# num_train = X_train.shape[0]
# num_test = X_test.shape[0]
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train.reshape(num_train, height, width, depth)
# X_test = X_test.reshape(num_test, height, width, depth)
X_train_gs = X_train_gs.astype('float32')
X_test_gs = X_test_gs.astype('float32')
X_train_gs /= 255 # Normalise data to [0, 1] range
X_test_gs /= 255 # Normalise data to [0, 1] range

y_train[y_train==10] = 0
y_test[y_test==10] = 0
# Y_train = np_utils.to_categorical(y_train) # One-hot encode the labels
# Y_test = np_utils.to_categorical(y_test) # One-hot encode the labels

model = Sequential()
model.add(Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu'))
model.add(Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format="channels_first"))
model.add(Dropout(drop_prob_1))
model.add(Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu'))
model.add(Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format="channels_first"))
model.add(Dropout(drop_prob_1))
model.add(Flatten())
model.add(Dense(hidden_size, activation='relu'))
model.add(Dropout(drop_prob_2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

model.fit(X_train_gs, y_train,
          batch_size=batch_size, nb_epoch=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation
model.evaluate(X_test_gs, y_test, verbose=1) # Evaluate the trained model on the test set!