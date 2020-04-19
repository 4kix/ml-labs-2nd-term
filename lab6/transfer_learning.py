import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import keras.backend.tensorflow_backend as tfback
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

# workaround for no available gpus exception
def _get_available_gpus():
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus

train_set = pd.read_csv('data/sign_mnist_train.csv')
test_set = pd.read_csv('data/sign_mnist_test.csv')

X_train = train_set.drop(['label'], axis=1).values
X_test = test_set.drop(['label'], axis=1).values

y_train = train_set.pop('label')
y_test = test_set.pop('label')

X_train = X_train.reshape(X_train.shape[0], 28, 28)
X_test = X_test.reshape(X_test.shape[0], 28, 28)

X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=2500)
X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=500)

IMAGE_RES = 224
def rescale_andnorm_dataset(dataset):
    set_length = dataset.shape[0]
    rescaled_dataset = np.zeros((set_length, IMAGE_RES, IMAGE_RES, 3))
    for i, img in enumerate(dataset):
        stacked_img = np.stack((img,)*3, axis=-1)
        stacked_img = tf.image.resize(stacked_img, (IMAGE_RES, IMAGE_RES))
        rescaled_dataset[i, :, :, :] = stacked_img
    return rescaled_dataset


X_train = X_train/255
X_test = X_test/255

X_train = rescale_andnorm_dataset(X_train)
X_test = rescale_andnorm_dataset(X_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)


BATCH_SIZE = 32
MODEL_URL = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2'

def rescale_and_norm_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label


URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))

feature_extractor.trainable = False

model = tf.keras.Sequential([
  feature_extractor,
  tf.keras.layers.Dense(25)
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

num_epochs = 10
fit_data = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=num_epochs,
                     validation_data=(X_val, y_val), verbose=1)

test_data = model.evaluate(X_test, y_test, verbose=2)

print('Test accuracy: {:.4f}'.format(test_data[1]))


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
