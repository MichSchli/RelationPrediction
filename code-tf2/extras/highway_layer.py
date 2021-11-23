import numpy as np
import tensorflow as tf
from model import Model
from common.shared_functions import glorot_variance, make_tf_variable, make_tf_bias

class HighwayLayer(Model):
    vertex_embedding_function = {'train': None, 'test': None}

    def __init__(self, shape, next_component=None, next_component_2=None):
        self.next_component = next_component
        self.next_component_2 = next_component_2
        self.shape = shape

    def compute_vertex_embeddings(self, mode='train'):
        if self.vertex_embedding_function[mode] is None:
            code_1 = self.next_component.get_all_codes(mode=mode)[0]
            code_2 = self.next_component_2.get_all_codes(mode=mode)[0]

            gates = self.get_gates(mode=mode)

            self.vertex_embedding_function[mode] = gates * code_1 + (1-gates) * code_2

        return self.vertex_embedding_function[mode]

    def local_initialize_train(self):
        variance = glorot_variance(self.shape)

        self.W = make_tf_variable(0, variance, self.shape)
        self.b = make_tf_bias(self.shape[1], init=1)

    def local_get_weights(self):
        return [self.W, self.b]

    def get_gates(self, mode='train'):
        code = self.next_component_2.get_all_codes(mode=mode)[0]
        hidden = tf.matmul(code, self.W) + self.b

        return tf.nn.sigmoid(hidden)

    def get_all_codes(self, mode='train'):
        collected_messages = self.compute_vertex_embeddings(mode=mode)

        return collected_messages, None, collected_messages

    def get_all_subject_codes(self, mode='train'):
        return self.compute_vertex_embeddings(mode=mode)

    def get_all_object_codes(self, mode='train'):
        return self.compute_vertex_embeddings(mode=mode)
