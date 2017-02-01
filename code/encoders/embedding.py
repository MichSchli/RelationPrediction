import numpy as np
import tensorflow as tf
from model import Model


class Embedding(Model):
    embedding_width = None

    W_embedding = None

    def __init__(self, settings, next_component=None):
        Model.__init__(self, next_component, settings)

    def parse_settings(self):
        self.embedding_width = int(self.settings['CodeDimension'])

    def local_initialize_train(self):
        vertex_matrix_shape = (self.entity_count, self.embedding_width)
        glorot_var_combined = np.sqrt(3 / (vertex_matrix_shape[0] + vertex_matrix_shape[1]))

        embedding_initial = np.random.normal(0, glorot_var_combined, size=vertex_matrix_shape).astype(np.float32)
        #np.random.randn(self.entity_count, self.embedding_width).astype(np.float32)

        self.W_embedding = tf.Variable(embedding_initial)

    def local_get_weights(self):
        return [self.W_embedding]

    def get_all_subject_codes(self, mode='train'):
        return self.W_embedding

    def get_all_object_codes(self, mode='train'):
        return self.W_embedding

    def get_all_codes(self, mode='train'):
        return self.W_embedding, None, self.W_embedding
