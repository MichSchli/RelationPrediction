import numpy as np
import tensorflow as tf
from model import Model

from common.shared_functions import glorot_variance, make_tf_variable, make_tf_bias

class RandomEmbedding(Model):
    embedding_width = None

    W = None
    b = None
    shape = None

    stored_W = None

    def __init__(self, shape, settings, next_component=None):
        Model.__init__(self, next_component, settings)
        self.shape = shape

    def get_all_codes(self, mode='train'):
        if self.stored_W is None:
            print("Warning: Vertices embedded as random vectors drawn U(-1,1)")
            self.stored_W = tf.random_uniform(self.shape, -1, 1, dtype=tf.float32)

        return self.stored_W, None, self.stored_W