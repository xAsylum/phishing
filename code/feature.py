from abc import ABC, abstractmethod
import numpy as np

from utils import params_count


def sgn(x):
    return 1 if x > 0 else -1


# This is an interface which I stopped development on, due to poor performance of
# SVM with long vectors of features

# ---- Interface ----

class FeatureFunction(ABC):
    @abstractmethod
    def phi(self, x: np.ndarray, out: np.ndarray, offset: int = 0) -> np.ndarray:
        pass

    def output_size(self, input_dim: int) -> int:
        """Return output dimension given input size."""
        raise NotImplementedError


# ---- Feature Functions ----

class Linear(FeatureFunction):
    def phi(self, x: np.ndarray, out: np.ndarray, offset: int = 0):
        x = np.asarray(x)
        out[offset] = 1.0
        out[offset + 1 : offset + 1 + len(x)] = x

    def output_size(self, input_dim: int) -> int:
        return input_dim + 1

class Zeros(FeatureFunction):
    def __init__(self, zeros_mask):
        """Use this function to zero out certain features"""
        self.take = [i for i, z in enumerate(zeros_mask) if z != 0]

    def phi(self, x: np.ndarray, out: np.ndarray, offset: int = 0):
        x = np.asarray(x)
        out[offset] = 1.0
        for i, idx in enumerate(self.take):
            out[offset + 1 + i] = x[idx]

    def output_size(self, input_dim: int) -> int:
        return len(self.take) + 1

class Quadratic(FeatureFunction):
    def phi(self, x: np.ndarray, out: np.ndarray, offset: int = 0):
        x = np.asarray(x)
        idx = offset
        for i in range(len(x)):
            for j in range(i, len(x)):
                out[idx] = x[i] * x[j]
                idx += 1

    def output_size(self, input_dim: int) -> int:
        return input_dim * (input_dim + 1) // 2

# ---- Combiner ----

class FeatureFunctionGenerator:
    @staticmethod
    def get_feature_function(func_type: type, params: dict) -> FeatureFunction:
        return func_type(**params)

    @staticmethod
    def build_pipeline(functions: list):
        """
        functions: list of tuples like [(Linear, {}),...]
        """
        input_dim = params_count
        instances = [FeatureFunctionGenerator.get_feature_function(f_type, f_params) for f_type, f_params in functions]
        sizes = [f.output_size(input_dim) for f in instances]
        offsets = np.cumsum([0] + sizes[:-1])
        total_size = sum(sizes)

        def feature_function(x):
            out = np.empty(total_size, dtype=np.float64)
            for f, offset in zip(instances, offsets):
                f.phi(x, out, offset)
            return out

        return feature_function, total_size

