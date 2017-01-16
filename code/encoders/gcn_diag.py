import numpy as np
import tensorflow as tf
from model import Model


class DiagGcn(Model):
    V_proj = None
    E_proj = None
    onehot_input = True
    vertex_embedding_function = {'train':None, 'test':None}

    def __init__(self, settings, graph_representation, next_component=None, onehot_input=False):
        Model.__init__(self, next_component, settings)
        self.graph_representation = graph_representation
        self.onehot_input = onehot_input

    def parse_settings(self):
        self.embedding_width = int(self.settings['InternalEncoderDimension'])
        self.dropout_keep_probability = float(self.settings['DropoutKeepProbability'])

    def local_initialize_train(self):
        vertex_feature_dimension = self.entity_count if self.onehot_input else self.embedding_width
        type_matrix_shape = (self.relation_count *2 + 1, self.embedding_width)
        vertex_matrix_shape = (vertex_feature_dimension, self.embedding_width)

        glorot_var_combined = np.sqrt(3/(vertex_matrix_shape[0] + vertex_matrix_shape[1]))
        type_init_var = 1

        vertex_projection_sender_initial = np.random.normal(0, glorot_var_combined, size=vertex_matrix_shape).astype(np.float32)
        type_initial = np.random.normal(0, type_init_var, size=type_matrix_shape).astype(np.float32)

        message_bias = np.zeros(self.embedding_width).astype(np.float32)

        self.V_proj_sender = tf.Variable(vertex_projection_sender_initial)
        self.V_types = tf.Variable(type_initial)
        self.B_message = tf.Variable(message_bias)

    def local_get_weights(self):
        return [self.V_proj_sender, self.V_types,
                self.B_message]

    def dot_or_lookup(self, features, weights):
        if self.onehot_input:
            return tf.nn.embedding_lookup(weights, features)
        else:
            return tf.matmul(features, weights)

    def get_vertex_features(self, mode='train'):
        sender_index_vector = self.get_graph().get_sender_indices()
        if self.onehot_input:
            return sender_index_vector
        else:
            code = tf.nn.dropout(self.next_component.get_all_codes(mode=mode)[0], self.dropout_keep_probability)
            sender_codes = tf.nn.embedding_lookup(code, sender_index_vector)

            return sender_codes

    def get_all_codes(self, mode='train'):
        collected_messages = self.compute_vertex_embeddings(mode=mode)

        return collected_messages, None, collected_messages

    def compute_vertex_embeddings(self, mode='train'):
        if self.vertex_embedding_function[mode] is None:
            sender_features = self.get_vertex_features(mode=mode)
            messages = self.compute_messages(sender_features)
            self.vertex_embedding_function[mode] = self.collect_messages(messages)

        return self.vertex_embedding_function[mode]

    def compute_messages(self, sender_features):
        sender_terms = self.dot_or_lookup(sender_features, self.V_proj_sender)
        message_types = self.get_graph().get_type_indices()
        type_diags = tf.nn.embedding_lookup(self.V_types, message_types)

        terms = tf.mul(sender_terms, type_diags)

        messages = tf.nn.relu(terms + self.B_message)
        return messages

    def collect_messages(self, messages):
        '''
        mtr_indices, mtr_values, mtr_shape = self.graph_representation.compute_sparse_mtr()

        mtr = tf.SparseTensor(indices=mtr_indices,
                              values=mtr_values,
                              shape=mtr_shape)
        '''

        mtr = self.get_graph().incidence_matrix(normalization=('global', 'recalculated'))

        collected_messages = tf.sparse_tensor_dense_matmul(mtr, messages)

        return collected_messages

    def get_all_subject_codes(self, mode='train'):
        return self.compute_vertex_embeddings(mode=mode)

    def get_all_object_codes(self, mode='train'):
        return self.compute_vertex_embeddings(mode=mode)