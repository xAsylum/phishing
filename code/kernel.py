from enum import Enum
import numpy as np
class KernelType(Enum):
    linear=1,
    polynomial=2,
    gaussian=3,
    sigmoid=4

class KernelFunctionGenerator:
    @staticmethod
    def get_kernel_function(kernel_type: KernelType, params):
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
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

    @staticmethod
    def combine(kernels: list):
        """
        kernels: list of tuples like (KernelType.linear, params_dict)
        """
        return lambda x, y: sum(
            KernelFunctionGenerator.get_kernel_function(kernel_type, params)(x, y)
            for kernel_type, params in kernels
        )