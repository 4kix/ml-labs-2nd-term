from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
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
num_epochs = 10
kernel_size = 3
pool_size = 2
conv_depth_1 = 6
conv_depth_2 = 6

num_classes = 10


# Load datasets
X_train = np.load('../lab1/X_train.npy')
y_train = np.load('../lab1/y_train.npy')

X_test = np.load('../lab1/X_test.npy')
y_test = np.load('../lab1/y_test.npy')

X_val = np.load('../lab1/X_val.npy')
y_val = np.load('../lab1/y_val.npy')

num_labels = 10

img_size = 28
num_train, _, _ = X_train.shape
num_test, _, _ = X_test.shape
num_val, _, _ = X_val.shape

X_train = X_train.reshape(num_train, img_size, img_size, 1)
X_test = X_test.reshape(num_test, img_size, img_size, 1)
X_val = X_val.reshape(num_val, img_size, img_size, 1)

X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_val = np.pad(X_val, ((0,0),(2,2),(2,2),(0,0)), 'constant')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
y_val = np_utils.to_categorical(y_val, num_classes)

model = Sequential()
model.add(Conv2D(6, 3, 3, border_mode='same', activation='relu', input_shape=(32, 32, 1)))
model.add(AveragePooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Conv2D(6, 3, 3, border_mode='same', activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

fit_data = model.fit(X_train, y_train,
          batch_size=batch_size, nb_epoch=num_epochs,
          verbose=2, validation_data=(X_val, y_val))
test_data = model.evaluate(X_test, y_test, verbose=2)
model.save("cnn.model_lenet.h5")


print('Test accuracy: {:.4f}'.format( test_data[1]))


# Plot training & validation accuracy values
plt.plot(fit_data.history['accuracy'])
plt.plot(fit_data.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(fit_data.history['loss'])
plt.plot(fit_data.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
