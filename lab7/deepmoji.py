#!git clone https://github.com/bfelbo/DeepMoji.git

import sys

sys.path.extend(['/content/DeepMoji/', '/content/DeepMoji/deepmoji'])
%cd /content/DeepMoji


%tensorflow_version 2.x
import tensorflow as tf

from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
# from keras_self_attention import AttentionWeightedAverage
from keras.datasets import imdb
from deepmoji.model_def import deepmoji_architecture
from deepmoji.attlayer import AttentionWeightedAverage

np.random.seed(1337)

nb_tokens = 20000
maxlen = 80
batch_size = 32

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=nb_tokens)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = deepmoji_architecture(nb_classes=2, nb_tokens=nb_tokens, maxlen=maxlen)
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
fit_data = model.fit(X_train, y_train, batch_size=batch_size, epochs=5,
          validation_data=(X_test, y_test))
test_data = model.evaluate(X_test, y_test, batch_size=batch_size)

print('Test accuracy: {:.4f}'.format(test_data[1]))

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
