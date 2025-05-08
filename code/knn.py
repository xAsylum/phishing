from utils import load_file
import numpy as np

class KNN:
    def __init__(self, path, train_frac = 0.6, valid_frac = 0.2, k = 1, distance_function = None):
        self.k = k
        self.train, self.valid, self.test = load_file(path, train_frac, valid_frac)
        self.distance = distance_function

    def measure_valid(self):
        return np.mean([self.predict_one(x) == y for (x, y) in self.valid])

    def check_accuracy(self):
        return np.mean([self.predict_one(x) == y for (x, y) in self.valid])

    def estimate_k(self):
        accuracy = np.float64(0)
        k = -1
        for test_k in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, len(self.train)]:
            if test_k > len(self.train):
                continue
            self.k = test_k
            measured = self.measure_valid()
            if accuracy < measured:
                accuracy = measured
                k = test_k
        self.k = k

    def predict_one(self, x):
        #todo: write predict_one
        pass

    def decide(self, p):
        return p >= 0.5 # agree if at least 50% integrity



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
                assert len(x) == len(y)
                x = np.array(x)
                y = np.array(y)
                return np.linalg.norm(x - y, ord=arg)

            return euclidean
        return None
