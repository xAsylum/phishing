from utils import load_file

class SVM:
    def __init__(self, path, train_frac = 0.6, valid_frac = 0.2):
        self.lr = 0.01
        self.train, self.valid, self.test = load_file(path, train_frac, valid_frac)

    def fit(self):
        pass

class KernelFunction:
    def __init__(self, kernel):
        pass