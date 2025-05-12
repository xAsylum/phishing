from utils import load_file, sign

class SVM:
    def __init__(self, path, train_frac = 0.6, valid_frac = 0.2,
                 kernel = KernelFunctionGenerator.get_kernel_function(KernelType.linear)):
        self.eta = 0.01
        self.epochs = 1000
        self.train, self.valid, self.test = load_file(path, train_frac, valid_frac)
        self.kernel_function = kernel
        self.w = np.zeros(31)
        self.C = 100

    def loss_one(self, x, y):
        X = np.array([1, *x])
        res = self.C * max(0, 1 -  y * self.kernel_function(self.w, X))
        return res
    def gradient_one(self, x, y):
        X = np.array([1, *x])
        if y * self.kernel_function(self.w, X) >= 1:
            return np.zeros(self.W.shape)
        return -y * X

    def L2gradient(self):
        res = np.mean([self.gradient_one(x, y)
                       for x, y in self.train])
        return res + self.w

    def fit(self):
        self.w = np.zeros(31)
        for _ in range(self.epochs):
            self.w -= self.eta * self.L2gradient()


    def accuracy(self):
        correct = 0
        for (x, y) in self.test:
            X = np.array([1, *x])
            correct += (sign(self.kernel_function(self.w, X)) == y)

        return correct / len(self.test)


class KernelType(Enum):
    linear=1,
    polynomial=2,
    gaussian=3,
    sigmoid=4

class KernelFunctionGenerator:
    @staticmethod
    def get_kernel_function(kernel_type: KernelType, **params):
        if kernel_type == KernelType.linear:
            return lambda x, y: np.dot(x, y)
        elif kernel_type == KernelType.polynomial:
            degree = params.get('degree', 3)
            coef0 = params.get('coef', 1)
            return lambda x, y: np.power((np.dot(x, y) + coef0), degree)
        elif kernel_type == KernelType.gaussian:
            gamma = params.get('gamma', 1.0)
            return lambda x, y: np.exp(-gamma * np.linalg.norm(x - y) ** 2)
        elif kernel_type == KernelType.sigmoid:
            return lambda x, y: 1 / (1 + np.exp(-np.dot(x, y)))