from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils

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



batch_size = 32
num_epochs = 20
kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512

num_train = 60000
num_test = 10000

height, width, depth = 28, 28, 1
num_classes = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(num_train, height, width, depth)
X_test = X_test.reshape(num_test, height, width, depth)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255 # Normalise data to [0, 1] range
X_test /= 255 # Normalise data to [0, 1] range

Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

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

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

model.fit(X_train, Y_train,
          batch_size=batch_size, nb_epoch=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation
model.evaluate(X_test, Y_test, verbose=1) # Evaluate the trained model on the test set!