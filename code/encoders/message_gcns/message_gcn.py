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

    def __init__(self, shape, settings, next_component=None, onehot_input=False, use_nonlinearity=True):
        self.onehot_input = onehot_input
        self.use_nonlinearity = use_nonlinearity
        self.shape = shape
        Model.__init__(self, next_component, settings)

    def needs_graph(self):
        return True

    '''
    Methods for message GCN:
    '''

    def get_vertex_features(self, senders=True, mode='train'):
        if senders:
            index_vector = self.get_graph().get_sender_indices()
        else:
            index_vector = self.get_graph().get_receiver_indices()

        if self.onehot_input:
            return index_vector
        else:
            #At the moment we use message level dropout, so disabled here
            #code = tf.nn.dropout(self.next_component.get_all_codes(mode=mode)[0], self.dropout_keep_probability)
            code = self.next_component.get_all_codes(mode=mode)[0]
            vertex_codes = tf.nn.embedding_lookup(code, index_vector)

            return vertex_codes

    def get_all_codes(self, mode='train'):
        collected_messages = self.compute_vertex_embeddings(mode=mode)

        return collected_messages, None, collected_messages

    def compute_vertex_embeddings(self, mode='train'):
        if self.vertex_embedding_function[mode] is None:
            sender_features = self.get_vertex_features(senders=True, mode=mode)
            receiver_features = self.get_vertex_features(senders=False, mode=mode)

            forward_messages, backward_messages = self.compute_messages(sender_features, receiver_features)
            if self.onehot_input:
                self_loop_messages = self.compute_self_loop_messages(tf.range(self.entity_count))
            else:
                self_loop_messages = self.compute_self_loop_messages(self.next_component.get_all_codes(mode=mode)[0])

            if mode == 'train':
                # We do "permanent" edge dropouts, so between layers only self-loops are dropped
                #forward_messages = tf.nn.dropout(forward_messages, self.dropout_keep_probability)
                #backward_messages = tf.nn.dropout(backward_messages, self.dropout_keep_probability)
                self_loop_messages = tf.nn.dropout(self_loop_messages, self.dropout_keep_probability)

            if self.onehot_input:
                self.vertex_embedding_function[mode] = self.combine_messages(forward_messages,
                                                                             backward_messages,
                                                                             self_loop_messages,
                                                                             tf.range(self.entity_count),
                                                                             mode=mode)
            else:
                self.vertex_embedding_function[mode] = self.combine_messages(forward_messages,
                                                                             backward_messages,
                                                                             self_loop_messages,
                                                                             self.next_component.get_all_codes(mode=mode)[0],
                                                                             mode=mode)

        return self.vertex_embedding_function[mode]


    def get_all_subject_codes(self, mode='train'):
        return self.compute_vertex_embeddings(mode=mode)

    def get_all_object_codes(self, mode='train'):
        return self.compute_vertex_embeddings(mode=mode)

