from enum import Enum

from numpy.random import shuffle

from utils import load_file, accuracy
import numpy as np

class DistanceFunction(Enum):
    Hamming=1,
    Euclidean=2

class DistanceFunctionGenerator:
    @staticmethod
    def distance_function(type, arg = None):
        if type == DistanceFunction.Hamming:
            def hamming(x, y):
                assert len(x) == len(y)
                x = np.array(x)
                y = np.array(y)
                return np.sum(x!=y)

            return hamming
        if type == DistanceFunction.Euclidean:
            def euclidean(x, y):
                assert(arg is not None)
                assert len(x) == len(y)
                x = np.array(x)
                y = np.array(y)
                return np.linalg.norm(x - y, ord=arg)

            return euclidean
        return None


def decide(p):
    if p >= 0:
        return 1 # agree if at least 50% integrity
    return -1


class KNN:
    def __init__(self, path, train_frac = 0.6, valid_frac = 0.2, k = 1, distance_function = DistanceFunctionGenerator.distance_function(DistanceFunction.Hamming)):
        self.valid_test = None
        self.k = k
        self.train_set, self.valid, self.test = load_file(path, train_frac, valid_frac)
        self.train = self.train_set
        self.distance = distance_function

    def measure_valid(self):
        return np.mean([self.predict_one(x) == y for (x, y) in self.valid])

    def check_accuracy(self):
        Y = [y for (_, y) in self.test]
        Y_pred = [self.predict_one(x) for (x, _) in self.test]
        return accuracy(Y, Y_pred)

    def estimate_k(self): #estimate optimal
        accuracy = np.float64(0)
        self.valid_test = []
        k = -1
        for test_k in [1, 2, 5, 10, 20, 50, 100, 150]:
            if test_k > len(self.train):
                continue
            self.k = test_k
            measured = self.measure_valid()
            self.valid_test.append((test_k, measured))
            if accuracy < measured:
                accuracy = measured
                k = test_k
        self.k = k

    def select_fraction(self, frac):
        negative = [x for x in self.train_set if x[1] == -1]
        positive = [x for x in self.train_set if x[1] == 1]
        neg_split = int(len(negative) * frac)
        pos_split = int(len(positive) * frac)
        train_set = ([x for x in negative[:neg_split]]
                     + [x for x in positive[:pos_split]])
        shuffle(train_set)
        self.train = train_set

    def predict_one(self, x):
        sorted_items = sorted(self.train, key=lambda y: self.distance(x, y[0]))[:self.k]
        return decide(np.sum([item[1] for item in sorted_items]))
