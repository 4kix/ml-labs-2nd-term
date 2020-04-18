import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load datasets
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

plt.style.use('ggplot')
np.random.seed(42)
train_sizes = [50, 100, 1000, 50000, 100000, 200000]

test_scores = []
for train_size in train_sizes:
    indices = np.random.randint(0, X_train.shape[0], train_size)

    X = X_train[indices, :, :] \
        .reshape(-1, X_train.shape[1] * X_train.shape[2])
    y = y_train[indices]

    clf = (LogisticRegression(random_state=10, solver='lbfgs', multi_class='multinomial')
           .fit(X, y))

    y_pred = clf.predict(X_test.reshape(X_test.shape[0], -1))

    test_score = accuracy_score(y_pred, y_test)
    test_scores.append(test_score)


plt.figure(figsize=(10, 7))
plt.xlabel('Training set size', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
for x, y in zip(train_sizes, test_scores):
    plt.text(x + 50, y, '{:.2f}'.format(y))

plt.scatter(train_sizes, test_scores, label='Test score', color='green')
plt.legend(loc=4)
plt.title('Accuracy dependency on train set size', fontsize=25)
plt.show()

