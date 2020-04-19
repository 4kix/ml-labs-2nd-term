from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from numpy import save
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
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

# Load data

folder = 'data/train/'

def load_dataset():
    X = np.zeros((25000, 100, 100, 3))
    y = np.zeros(25000)
    counter = 0
    for file in listdir(folder):
        label = 0.0
        if file.startswith('cat'):
            label = 1.0
        photo = load_img(folder + file, target_size=(100, 100))
        photo = img_to_array(photo)
        X[counter, :, :, :] = photo
        y[counter] = label
        counter+=1
    print(X.shape, y.shape)
    save('data/photos.npy', X)
    save('data/labels.npy', y)
    return X, y

X_train, y_train = load_dataset()


# split on train, test, val sets

# X_train = load('data/photos.npy')
# y_train = load('data/labels.npy')

def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)
X_train = rgb2gray(X_train)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.6)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

# CNN
batch_size = 64
num_epochs = 30
kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512

model = Sequential()
model.add(Conv2D(conv_depth_1, (kernel_size, kernel_size), border_mode='same', activation='relu', input_shape=(100, 100, 1)))
model.add(Conv2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format="channels_first"))
model.add(Conv2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu'))
model.add(Conv2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format="channels_first"))
model.add(Flatten())
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

fit_data = model.fit(X_train, y_train,
          batch_size=batch_size, nb_epoch=num_epochs,
          verbose=1, validation_data=(X_val, y_val))
score = model.evaluate(X_test, y_test, verbose=2)
print('Test accuracy: {:.4f}', score[1])
print('Test loss: {:.4f}', score[0])


plt.plot(fit_data.history['accuracy'])
plt.plot(fit_data.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


plt.plot(fit_data.history['loss'])
plt.plot(fit_data.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save('dogs_vs_cats_model.h5')