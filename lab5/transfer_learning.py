import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import keras.backend.tensorflow_backend as tfback


print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

# workaround for no available gpus exception
def _get_available_gpus():
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus


(train_examples, validation_examples, test_examples), info = tfds.load(
    'cats_vs_dogs',
    with_info=True,
    as_supervised=True,
    split=[
        tfds.Split.TRAIN.subsplit(tfds.percent[:80]),
        tfds.Split.TRAIN.subsplit(tfds.percent[80:90]),
        tfds.Split.TRAIN.subsplit(tfds.percent[90:])
    ],
)

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes

IMAGE_RES = 224
BATCH_SIZE = 32
MODEL_URL = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2'

def rescale_and_norm_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label


train_batches = train_examples.shuffle(num_examples//4).map(rescale_and_norm_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(rescale_and_norm_image).batch(BATCH_SIZE).prefetch(1)
test_batches = test_examples.map(rescale_and_norm_image).batch(BATCH_SIZE).prefetch(1)


URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))

feature_extractor.trainable = False

model = tf.keras.Sequential([
  feature_extractor,
  tf.keras.layers.Dense(2)
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

num_epochs = 6
fit_data = model.fit(train_batches, epochs=num_epochs,
                    validation_data=validation_batches, verbose=1)

test_data = model.evaluate(test_batches, verbose=2)

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
