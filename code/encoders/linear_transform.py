import numpy as np
import tensorflow as tf
from model import Model


class LinearTransform(Model):
    embedding_width = None

    W_embedding = None

    def parse_settings(self):
        self.embedding_width = int(self.settings['EmbeddingWidth'])

    def local_initialize_train(self):
        transform_matrix = np.random.normal(0, np.sqrt(1 / (2 * self.embedding_width)),
                                                 size=(self.embedding_width, self.embedding_width)).astype(np.float32)

        transform_bias = np.zeros(self.embedding_width).astype(np.float32)

        self.W = tf.Variable(transform_matrix)
        self.b = tf.Variable(transform_bias)

    def local_get_weights(self):
        return [self.W, self.b]

    def get_all_subject_codes(self, mode='train'):
        code = self.next_component.get_all_subject_codes(mode=mode)
        return tf.matmul(code, self.W) + self.b

    def get_all_object_codes(self, mode='train'):
        code = self.next_component.get_all_object_codes(mode=mode)
        return tf.matmul(code, self.W) + self.b

    def get_all_codes(self, mode='train'):
        codes = self.next_component.get_all_codes(mode=mode)

        return tf.matmul(codes[0], self.W) + self.b, codes[1], tf.matmul(codes[2], self.W) + self.b
