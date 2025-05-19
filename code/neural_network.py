import numpy as np
from numpy.ma.core import argmax
from numpy.random import shuffle

from utils import load_file, Dataset


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)


class NeuralNetwork:
    def __init__(self, path, architecture, train_frac=0.6, valid_frac=0.2):
        self.train_set, self.valid, self.test = load_file(path, train_frac, valid_frac)
        self.train = self.train_set

        # Architecture: input layer (31), hidden layers, output layer (2)
        self.architecture = [31] + architecture + [2]
        self.L = len(self.architecture)
        self.weights = []
        self.biases = []

        self.W = {}  # Weight matrices
        self.b = {}  # Bias vectors

        self.lc = [] # Learning Curve

        for l in range(self.L - 1):
            input_dim = self.architecture[l]
            output_dim = self.architecture[l + 1]
            self.W[l + 1] = (np.random.randn(output_dim, input_dim)
                         * np.sqrt(2. / input_dim))
            self.b[l + 1] = np.zeros((output_dim, 1))

    def forwardpropagation(self, x):
        a = {0: x}
        z = {}
        for l in range(self.L - 1):
            z[l + 1] = self.W[l + 1].dot(a[l]) + self.b[l + 1]
            a[l + 1] = sigmoid(z[l + 1])
        return a, z

    def backpropagation(self, x, y):
        a, z = self.forwardpropagation(x)
        dW = {}
        db = {}
        delta = {}

        L = self.L - 1  # Total number of layers - 1 for zero-based indexing

        delta[L] = (a[L] - y)
        dW[L] = np.dot(delta[L], z[L - 1].T)
        db[L] = delta[L]
        for l in reversed(range(1, L)):
            delta[l] = np.dot(self.W[l + 1].T, delta[l + 1]) * sigmoid_derivative(a[l])
            dW[l] = np.dot(delta[l], a[l - 1].T)
            db[l] = delta[l]

        return dW, db

    def fit(self, epochs=1000, lr=0.01):
        m = len(self.train)
        self.lc = []
        for epoch in range(epochs):
            total_loss = 0

            dW_sum = [np.zeros_like(self.W[l]) for l in self.W]
            db_sum = [np.zeros_like(self.b[l]) for l in self.b]

            for i in range(m):
                (x, y) = self.train[i]
                if y == -1:
                    y = np.array([[1], [0]])
                else:
                    y = np.array([[0], [1]])
                x = [1, *x]
                x = np.array(x).reshape(-1, 1)
                a, _ = self.forwardpropagation(x)
                loss = -np.dot(y.T, np.log(softmax(a[self.L - 1]))).item() # Cross-entropy
                total_loss += loss
                dW, db = self.backpropagation(x, y)

                for l in range(1, self.L):
                    dW_sum[l - 1] += dW[l]
                    db_sum[l - 1] += db[l]

            for l in range(1, self.L):
                self.W[l] -= lr * dW_sum[l - 1] / m
                self.b[l] -= lr * db_sum[l - 1] / m
            if epoch % 50 == 0:
                acc = self.predict(Dataset.Valid)
                self.lc.append((epoch, acc, total_loss / m))
                print(f"Epoch {epoch}: Accuracy ={acc}, Loss = {total_loss / m:.4f}")

    def select_fraction(self, frac):
        negative = [x for x in self.train_set if x[1] == -1]
        positive = [x for x in self.train_set if x[1] == 1]
        neg_split = int(len(negative) * frac)
        pos_split = int(len(positive) * frac)
        train_set = ([x for x in negative[:neg_split]]
                     + [x for x in positive[:pos_split]])
        shuffle(train_set)
        self.train = train_set

    def predict(self, dataset: Dataset = Dataset.Test):
        data = None
        if dataset == Dataset.Test:
            data = self.test
        else:
            data = self.valid
        m = len(data)
        count = 0
        for i in range(m):
            (x, y) = data[i]
            if y == -1:
                y = 0
            x = [1, *x]
            x = np.array(x).reshape(-1, 1)
            a, z = self.forwardpropagation(x)
            y_pred = np.argmax(a[self.L - 1])
            count += (y == y_pred)
        count /= m
        return count


