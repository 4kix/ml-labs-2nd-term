import matplotlib.pyplot as plt
# import keras
import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
# export CUDA_VISIBLE_DEVICES=0
import tensorflow as tf
import tensorflow_datasets as tfds
# from keras.backend.tensorflow_backend import set_session

# tf.config.experimental.set_visible_devices(physical_devices[1:], 'GPU')


# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# gpu_options.per_process_gpu_memory_fraction = 0.85
# physical_devices = tf.config.list_physical_devices('GPU')
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.85
physical_devices = tf.config.list_physical_devices()
print(physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.InteractiveSession(config=config)
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# tf.config.gpu.set_per_process_memory_growth(True)



ds, info = tfds.load('imdb_reviews/subwords8k',
                     with_info=True,
                     as_supervised=True)

train_examples, test_examples = ds['train'], ds['test']

encoder = info.features['text'].encoder

# print('Vocabulary size: {}'.format(encoder.vocab_size))

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = (train_examples
                 .shuffle(BUFFER_SIZE)
                 .padded_batch(BATCH_SIZE, padded_shapes=([None], [])))

test_dataset = (test_examples
                .padded_batch(BATCH_SIZE, padded_shapes=([None], [])))

# train_dataset = (train_examples
#                  .shuffle(BUFFER_SIZE)
#                  .padded_batch(BATCH_SIZE))

# test_dataset = (test_examples
#                 .padded_batch(BATCH_SIZE))

# RECCUR
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer="adam",
              metrics=['accuracy'])

model.summary()

history = model.fit(train_dataset, epochs=5,
                    validation_data=test_dataset,
                    validation_steps=30)

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec


def sample_predict(sample_pred_text, pad):
    encoded_sample_pred_text = encoder.encode(sample_pred_text)

    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

    return (predictions)


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
