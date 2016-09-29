import numpy as np
from theano import tensor as T

def glorot_initialization(n_from, n_to):
        value = np.sqrt(6)/ np.sqrt(n_from + n_to)
        return np.random.uniform(low=-value, high=value, size=(n_from, n_to)).astype(np.float32)

def apply_transform_to_elements(Vectors, Transformation):
        with_bias = T.concatenate([T.ones((Vectors.shape[0], 1), dtype=Vectors.dtype), Vectors], axis=1)
        return T.dot(with_bias, Transformation) 

def lecunn_tanh(v):
    return 1.7159*T.tanh(0.66667*v)
