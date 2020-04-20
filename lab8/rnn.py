import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read data
data = pd.read_csv('Sunspots.csv', parse_dates=['Date'], index_col=['Date'])

#fetch and format data
idx_timeline = data['index'].to_numpy()
series = data['Monthly Mean Total Sunspot Number'].to_numpy()

split = int(int(len(idx_timeline) * 0.7))
timeline_train = idx_timeline[:split]
sunspots_train = series[:split]
timeline_val = idx_timeline[split:]
sunspots_val = series[split:]


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda w: (w[:-1], w[1:]))
    return dataset.batch(batch_size).prefetch(1)

train_set = windowed_dataset(series, window_size=64, batch_size=100, shuffle_buffer=1000)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal",
                               activation="relu", input_shape=[None, 1]))
model.add(tf.keras.layers.LSTM(64, return_sequences=True))
model.add(tf.keras.layers.LSTM(64, return_sequences=True))
model.add(tf.keras.layers.Dense(8))
model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Lambda(lambda x: x * 400))

model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['mae'])
model.summary()

fit_data = model.fit(train_set, epochs=10, verbose=2)

dataset = tf.data.Dataset.from_tensor_slices(series[..., np.newaxis])
dataset = dataset.window(64, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda w: w.batch(64))
dataset = dataset.batch(32).prefetch(1)

predictions = model.predict(dataset)
predictions = predictions[split - 64:-1, -1, 0]

plt.figure(figsize=(10, 6))
plt.plot(timeline_val[0:None], sunspots_val[0:None], "-")
plt.plot(timeline_val[0:None], predictions[0:None], "-")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.show()