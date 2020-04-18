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
num_hidden_unit_1 = 1024
num_hidden_unit_2 = 512
num_hidden_unit_3 = 256
num_hidden_unit_4 = 128

learning_rate = 0.4
input_size = img_size * img_size
num_steps = 10000
lmbda = 1e-3

graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0)
    X_train_tf = tf.placeholder(tf.float32, shape=(input_size, batch_size))
    y_train_tf = tf.placeholder(tf.float32, shape=(num_labels, batch_size))
    X_val_tf = tf.constant(X_val)
    X_test_tf = tf.constant(X_test)

    W1 = tf.Variable(tf.truncated_normal([num_hidden_unit_1, input_size], stddev=np.sqrt(2.0 / input_size)))
    b1 = tf.Variable(tf.zeros([num_hidden_unit_1, 1]))
    W2 = tf.Variable(tf.truncated_normal([num_hidden_unit_2, num_hidden_unit_1], stddev=np.sqrt(2.0 / num_hidden_unit_1)))
    b2 = tf.Variable(tf.zeros([num_hidden_unit_2, 1]))
    W3 = tf.Variable(tf.truncated_normal([num_hidden_unit_3, num_hidden_unit_2], stddev=np.sqrt(2.0 / num_hidden_unit_2)))
    b3 = tf.Variable(tf.zeros([num_hidden_unit_3, 1]))
    W4 = tf.Variable(tf.truncated_normal([num_hidden_unit_4, num_hidden_unit_3], stddev=np.sqrt(2.0 / num_hidden_unit_3)))
    b4 = tf.Variable(tf.zeros([num_hidden_unit_4, 1]))
    W5 = tf.Variable(tf.truncated_normal([num_labels, num_hidden_unit_4], stddev=np.sqrt(2.0 / num_hidden_unit_4)))
    b5 = tf.Variable(tf.zeros([num_labels, 1]))

    Z1 = tf.matmul(W1, X_train_tf) + b1
    # A1 = tf.nn.dropout(tf.nn.relu(Z1), 0.25)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3
    A3 = tf.nn.relu(Z3)
    Z4 = tf.matmul(W4, A3) + b4
    A4 = tf.nn.relu(Z4)
    Z5 = tf.matmul(W5, A4) + b5

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.transpose(y_train_tf), logits=tf.transpose(Z5)))
    # loss += lmbda * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5))


    train_prediction = tf.nn.softmax(Z5, dim=0)
    y_val_prediction = tf.nn.softmax(tf.matmul(W5, tf.nn.relu(
        tf.matmul(W4, tf.nn.relu(
        tf.matmul(W3, tf.nn.relu(
        tf.matmul(W2, tf.nn.relu(
        tf.matmul(W1, X_val_tf) + b1)) + b2)) + b3)) + b4)) + b5, dim=0)
    y_test_prediction = tf.nn.softmax(
        tf.matmul(W5, tf.nn.relu(
        tf.matmul(W4, tf.nn.relu(
        tf.matmul(W3, tf.nn.relu(
        tf.matmul(W2, tf.nn.relu(
        tf.matmul(W1, X_test_tf) + b1)) + b2)) + b3)) + b4)) + b5, dim=0)

    learning_rate = tf.train.exponential_decay(0.5, global_step, 5000, 0.80, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()

    for step in tnrange(num_steps):

        offset = (step * batch_size) % (y_train.shape[1] - batch_size)
        batch_data = X_train[:, offset:(offset + batch_size)]
        batch_labels = y_train[:, offset:(offset + batch_size)]

        _, l, batch_Y_pred = session.run([optimizer, loss, train_prediction], feed_dict={X_train_tf: batch_data, y_train_tf: batch_labels})

        if (step % 200 == 0):
            print("Step #{}".format(step))
            print('Minibatch loss: {:.3f}. batch accuracy: {:.1f}%, Valid accuracy: {:.1f}%.' \
                  .format( l, compute_accuracy(batch_Y_pred, batch_labels), compute_accuracy(y_val_prediction.eval(), y_val)))

    print('Test accuracy: {:.1f}%'.format(compute_accuracy(y_test_prediction.eval(), y_test)))


