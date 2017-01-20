import numpy as np
import tensorflow as tf
from model import Model


class Embedding(Model):
    embedding_width = None

    W_embedding = None

    def __init__(self, settings):
        Model.__init__(self, None, settings)

    def parse_settings(self):
        self.embedding_width = int(self.settings['CodeDimension'])

    def local_initialize_train(self):
        embedding_initial = np.random.randn(self.entity_count, self.embedding_width).astype(np.float32)

        self.W_embedding = tf.Variable(embedding_initial)

    def local_get_weights(self):
        return [self.W_embedding]

    def get_all_subject_codes(self, mode='train'):
        return self.W_embedding

    def get_all_object_codes(self, mode='train'):
        return self.W_embedding

    def get_all_codes(self, mode='train'):
        return self.W_embedding, None, self.W_embedding
