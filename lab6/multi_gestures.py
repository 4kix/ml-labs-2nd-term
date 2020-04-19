import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
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

# loading data

train_set = pd.read_csv('data/sign_mnist_train.csv')
test_set = pd.read_csv('data/sign_mnist_test.csv')

X_train = train_set.drop(['label'], axis=1).values
X_test = test_set.drop(['label'], axis=1).values

y_train = train_set.pop('label')
y_test = test_set.pop('label')

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train/255
X_test = X_test/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)

#CNN
batch_size = 512
num_epochs = 10
kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512

model = Sequential()
model.add(Conv2D(conv_depth_1, (kernel_size, kernel_size), border_mode='same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format="channels_first"))
model.add(Conv2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format="channels_first"))
model.add(Flatten())
model.add(Dense(hidden_size, activation='relu'))
model.add(Dropout(drop_prob_2))
model.add(Dense(25, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

fit_data = model.fit(X_train, y_train,
          batch_size=batch_size, nb_epoch=num_epochs,
          verbose=2, validation_data=(X_val, y_val))
test_data = model.evaluate(X_test, y_test, verbose=2)

print('Test accuracy: {:.4f}'.format(test_data[1]))
print('Test loss: {:.4f}'.format(test_data[0]))


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

model.save('gestures_model.h5')
