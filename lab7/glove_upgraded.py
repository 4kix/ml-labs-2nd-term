from tqdm import tqdm
import numpy as np
import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import tensorflow as tf
import tensorflow_datasets as tfds

# from codecs import open

ds, info = tfds.load('imdb_reviews/subwords8k',
                     with_info=True,
                     as_supervised=True)

train_examples, test_examples = ds['train'], ds['test']

encoder = info.features['text'].encoder

UFFER_SIZE = 10000
BATCH_SIZE = 128

train_dataset = (train_examples
                 .shuffle(10000)
                 .padded_batch(BATCH_SIZE, padded_shapes=([None], [])))

test_dataset = (test_examples
                .padded_batch(BATCH_SIZE, padded_shapes=([None], [])))


GLOVE_DIR = 'glove6b/'
GLOVE_EMBEDDING_DIM = 100

unk_word_embedding = np.zeros(GLOVE_EMBEDDING_DIM)
embeddings_index = {}
with open(GLOVE_DIR + f'glove.6B.{GLOVE_EMBEDDING_DIM}d.txt', encoding='utf-8', errors='ignore') as glove_file:
    for i, line in tqdm(enumerate(glove_file)):
        word, *word_embedding = line.split()
        word_embedding = np.array(word_embedding, dtype='float32')
        embeddings_index[word] = word_embedding
        unk_word_embedding += word_embedding
    unk_word_embedding = unk_word_embedding / i

embedding_matrix = np.zeros((encoder.vocab_size, GLOVE_EMBEDDING_DIM))
for i, word in enumerate(encoder._subwords):
    embedding_vector = embeddings_index.get(word.rstrip('_'), unk_word_embedding)
    embedding_matrix[i] = embedding_vector

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, GLOVE_EMBEDDING_DIM),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer="adam",
              metrics=['accuracy'])

model.summary()

fit_data = model.fit(train_dataset, epochs=5,
                    validation_data=test_dataset,
                    validation_steps=30)

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
