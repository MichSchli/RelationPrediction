import numpy as np
import tensorflow as tf
from model import Model

class DropoverLayer(Model):
    vertex_embedding_function = {'train': None, 'test': None}

    def __init__(self, shape, next_component=None, next_component_2=None):
        self.next_component = next_component
        self.next_component_2 = next_component_2
        self.shape = shape

    def compute_vertex_embeddings(self, mode='train'):
        if self.vertex_embedding_function[mode] is None and mode=='train':
            code_1 = self.next_component.get_all_codes(mode=mode)[0]
            code_2 = self.next_component_2.get_all_codes(mode=mode)[0]

            choice = tf.random_uniform(self.shape, -1, 1, dtype=tf.float32)

            self.vertex_embedding_function[mode] = tf.where(choice > 0, x=code_1, y=code_2)
        elif mode=='test':
            self.vertex_embedding_function[mode] = self.next_component.get_all_codes(mode=mode)[0]

        return self.vertex_embedding_function[mode]


    def get_all_codes(self, mode='train'):
        collected_messages = self.compute_vertex_embeddings(mode=mode)

        return collected_messages, None, collected_messages

    def get_all_subject_codes(self, mode='train'):
        return self.compute_vertex_embeddings(mode=mode)

    def get_all_object_codes(self, mode='train'):
        return self.compute_vertex_embeddings(mode=mode)
