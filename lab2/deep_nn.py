import numpy as np
from tqdm import tnrange
import tensorflow.compat.v1 as tf


# Load datasets
X_train = np.load('../lab1/X_train.npy')
y_train = np.load('../lab1/y_train.npy')

X_test = np.load('../lab1/X_test.npy')
y_test = np.load('../lab1/y_test.npy')

X_val = np.load('../lab1/X_val.npy')
y_val = np.load('../lab1/y_val.npy')

img_size = 28
num_labels = 10


def flatten_dataset(X, y):
    X = (X.reshape((-1, img_size * img_size)).astype(np.float32)).T
    y = ((np.arange(num_labels) == y[:, None]).astype(np.float32)).T
    return X, y


def compute_accuracy(predictions, labels):
    return (np.sum(np.argmax(predictions, axis=0) == np.argmax(labels, axis=0))
            / labels.shape[1] * 100)

# Flatten datasets to fit model
X_train, y_train = flatten_dataset(X_train, y_train)
X_val, y_val = flatten_dataset(X_val, y_val)
X_test, y_test = flatten_dataset(X_test, y_test)
print('Training set: ', X_train.shape, y_train.shape)
print('Validation set: ', X_val.shape, y_val.shape)
print('Test set: ', X_test.shape, y_test.shape)

batch_size = 256
num_hidden_unit = 1024

graph = tf.Graph()
with graph.as_default():
    X_train_tf = tf.placeholder(tf.float32, shape=(img_size * img_size, batch_size))
    y_train_tf = tf.placeholder(tf.float32, shape=(num_labels, batch_size))
    X_val_tf = tf.constant(X_val)
    X_test_tf = tf.constant(X_test)

    W1 = tf.Variable(tf.truncated_normal([num_hidden_unit, img_size * img_size]))
    b1 = tf.Variable(tf.zeros([num_hidden_unit, 1]))
    W2 = tf.Variable(tf.truncated_normal([num_labels, num_hidden_unit]))
    b2 = tf.Variable(tf.zeros([num_labels, 1]))

    logits = tf.matmul(W2, tf.nn.relu(tf.matmul(W1, X_train_tf) + b1)) + b2
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.transpose(y_train_tf), logits=tf.transpose(logits)))

    train_prediction = tf.nn.softmax(logits, dim=0)
    valid_prediction = tf.nn.softmax(tf.matmul(W2, tf.nn.relu(tf.matmul(W1, X_val_tf) + b1)) + b2, dim=0)
    test_prediction = tf.nn.softmax(tf.matmul(W2, tf.nn.relu(tf.matmul(W1, X_test_tf) + b1)) + b2, dim=0)

    optimizer = tf.train.GradientDescentOptimizer(0.4).minimize(loss)

num_steps = 10000
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()

    for step in tnrange(num_steps):

        offset = (step * batch_size) % (y_train.shape[1] - batch_size)
        batch_data = X_train[:, offset:(offset + batch_size)]
        batch_labels = y_train[:, offset:(offset + batch_size)]

        feed_dict = {
            X_train_tf: batch_data,
            y_train_tf: batch_labels
        }

        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % 250 == 0):
            print("Step #{}".format(step))
            print('Batch loss: {:.3f}. batch acc: {:.1f}%, Valid acc: {:.1f}%.'
                  .format(l, compute_accuracy(predictions, batch_labels), compute_accuracy(valid_prediction.eval(), y_val)))

    print('Test acc: {:.1f}%'.format(compute_accuracy(test_prediction.eval(), y_test)))
