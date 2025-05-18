import numpy as np

from utils import load_file, sign, params_count, Dataset, accuracy
from kernel import *
from feature import *

class SVM:
    def __init__(self, path, train_frac = 0.6, valid_frac = 0.2):


        self.out_dim = params_count + 1

        self.base_function = lambda x : np.array([1, *x])
        self.eta = 0.025
        self.epochs = 2500
        self.train, self.valid, self.test = load_file(path, train_frac, valid_frac)
        self.w = np.zeros(params_count + 1)
        self.alpha = 0.0001
        self.reg_res = []
        self.lc = []

        # Kernel stuff
        self.kernel_vector = None
        self.alpha_kernel = None
        self.K = None
        self.kernel_function = KernelFunctionGenerator.combine([(KernelType.linear, {})])
        self.train_inputs = [x for x, _ in self.train]  # Save for prediction

    def set_kernel(self, kernel):
        self.kernel_function = kernel

    def set_base(self, base_function, out_dim):
        self.base_function = base_function
        self.w = np.zeros(out_dim)
        self.out_dim = out_dim

    def loss_one(self, x, y):
        X = self.base_function(x)
        res = max(0, 1 -  y * np.dot(self.w, X))
        return res

    def loss(self):
        return np.sum([self.loss_one(x, y) for (x, y) in self.train])

    def gradient_one(self, x, y):
        X = self.base_function(x)
        if y * np.dot(self.w, X) >= 1:
            return np.zeros(len(X))
        return -y * X

    def L2gradient(self):
        grad = np.zeros(len(self.w))
        for (x, y) in self.train:
            grad += self.gradient_one(x, y)
        grad /= len(self.train)
        grad += self.alpha * self.w
        grad[0] -= self.alpha * self.w[0]
        return grad  # Proper scaling

    def L2loss(self):
        return self.alpha * (np.dot(self.w, self.w) - self.w[0] ** 2) + self.loss()

    def fit(self, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        self.w = np.zeros(self.out_dim)
        self.lc = []
        eta_t = self.eta

        for epoch in range(self.epochs):
            gradient = eta_t * self.L2gradient()
            self.w = self.w - gradient

            #if epoch % 300 == 0:
            #    current_loss = self.L2loss()
            #    accuracy = self.accuracy(Dataset.Valid)
            #    self.lc.append((epoch, accuracy, current_loss))
            #    print(f"Epoch {epoch}, Loss: {current_loss:.6f}, LR: {eta_t:.6f}")


    def estimate_regularization(self):
        self.reg_res = []
        best_c = 0
        best_score = 0
        reg = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2]
        for c in reg:
            self.fit(c)
            score = self.accuracy(Dataset.Valid)
            if score > best_score:
                best_score = score
                best_c = c
            self.reg_res.append((c, score))
            print(c, score)
        self.fit(best_c)
        print(self.accuracy(Dataset.Test))


    def fit_kernel(self, alpha=None):
        if alpha is not None:
            self.alpha_kernel = alpha

        m = len(self.train)
        X = self.train_inputs
        Y= np.array([y for _, y in self.train])

        K = np.array([
            [self.kernel_function(np.array(X[i]), np.array(X[j])) for j in range(m)] for i in range(m)
        ])
        self.kernel_vector = np.linalg.solve(K + self.alpha_kernel * np.eye(m), Y)

    def predict_kernel(self, x):
        k_vec = np.array([self.kernel_function(x, x_train) for x_train in self.train_inputs])
        return np.dot(k_vec, self.kernel_vector)

    def kernel_accuracy(self, dataset: Dataset = Dataset.Test):
        data = None
        if dataset == Dataset.Test:
            data = self.test
        else:
            data = self.valid

        Y_pred = []
        Y = []
        for (x, y) in data:
            Y.append(y)
            if self.predict_kernel(np.array(x)) > 0:
                Y_pred.append(1)
            else:
                Y_pred.append(-1)

        return accuracy(Y_pred, Y)

    def accuracy(self, dataset: Dataset = Dataset.Test):
        data = None
        if dataset == Dataset.Test:
            data = self.test
        else:
            data = self.valid
        Y_pred = []
        Y = []
        for (x, y) in data:
            X = self.base_function(x)
            Y.append(y)
            if np.dot(self.w, X) > 0:
                Y_pred.append(1)
            else:
                Y_pred.append(-1)

        return accuracy(Y_pred, Y)

