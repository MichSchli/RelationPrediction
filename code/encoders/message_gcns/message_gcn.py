import numpy as np
import tensorflow as tf
from model import Model

'''
Abstract class representing a GCN based on summing and collecting messages. Derived classes
vary only in how messages are computed.
'''
class MessageGcn(Model):

    onehot_input = True
    use_nonlinearity = True
    vertex_embedding_function = {'train': None, 'test': None}

    def __init__(self, settings, graph_representation, next_component=None, onehot_input=False, use_nonlinearity=True):
        Model.__init__(self, next_component, settings)
        self.graph_representation = graph_representation
        self.onehot_input = onehot_input
        self.use_nonlinearity = use_nonlinearity

    def needs_graph(self):
        return True

    '''
    Methods for message GCN:
    '''

    def get_vertex_features(self, mode='train'):
        sender_index_vector = self.get_graph().get_sender_indices()
        if self.onehot_input:
            return sender_index_vector
        else:
            #At the moment we use message level dropout, so disabled here
            #code = tf.nn.dropout(self.next_component.get_all_codes(mode=mode)[0], self.dropout_keep_probability)
            code = self.next_component.get_all_codes(mode=mode)[0]
            sender_codes = tf.nn.embedding_lookup(code, sender_index_vector)

            return sender_codes

    def get_all_codes(self, mode='train'):
        collected_messages = self.compute_vertex_embeddings(mode=mode)

        return collected_messages, None, collected_messages

    def compute_vertex_embeddings(self, mode='train'):
        if self.vertex_embedding_function[mode] is None:
            sender_features = self.get_vertex_features(mode=mode)

            messages = self.compute_messages(sender_features)

            if mode == 'train':
                messages = tf.nn.dropout(messages, self.dropout_keep_probability)

            summed_messages = self.sum_messages(messages)

            if self.use_nonlinearity:
                summed_messages = tf.nn.relu(summed_messages)

            self.vertex_embedding_function[mode] = summed_messages

        return self.vertex_embedding_function[mode]


    def get_all_subject_codes(self, mode='train'):
        return self.compute_vertex_embeddings(mode=mode)

    def get_all_object_codes(self, mode='train'):
        return self.compute_vertex_embeddings(mode=mode)

    '''
    General methods, should be moved
    '''

    def dot_or_lookup(self, features, weights):
        if self.onehot_input:
            return tf.nn.embedding_lookup(weights, features)
        else:
            return tf.matmul(features, weights)