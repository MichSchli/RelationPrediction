import numpy as np
import tensorflow as tf
from model import Model


class BipartiteGcn(Model):
    onehot_input = True
    vertex_embedding_function = {'train':None, 'test':None}

    def __init__(self, settings, graph_representation, next_component=None):
        Model.__init__(self, next_component, settings)
        self.graph_representation = graph_representation
        self.onehot_input = next_component is None

    def parse_settings(self):
        self.embedding_width = int(self.settings['InternalEncoderDimension'])
        self.dropout_keep_probability = float(self.settings['DropoutKeepProbability'])

    def local_initialize_train(self):
        e_vertex_feature_dimension = self.entity_count if self.onehot_input else self.embedding_width
        r_vertex_feature_dimension = self.relation_count if self.onehot_input else self.embedding_width

        e_vertex_matrix_shape = (e_vertex_feature_dimension, self.embedding_width)
        r_vertex_matrix_shape = (r_vertex_feature_dimension, self.embedding_width)

        e_glorot_var = np.sqrt(3/(e_vertex_matrix_shape[0] + e_vertex_matrix_shape[1]))
        r_glorot_var = np.sqrt(3/(r_vertex_matrix_shape[0] + r_vertex_matrix_shape[1]))

        e_forward = np.random.normal(0, e_glorot_var, size=e_vertex_matrix_shape).astype(np.float32)
        e_backward = np.random.normal(0, e_glorot_var, size=e_vertex_matrix_shape).astype(np.float32)
        r_forward = np.random.normal(0, r_glorot_var, size=r_vertex_matrix_shape).astype(np.float32)
        r_backward = np.random.normal(0, r_glorot_var, size=r_vertex_matrix_shape).astype(np.float32)

        bias_init = np.zeros(self.embedding_width).astype(np.float32)

        self.E_forward = tf.Variable(e_forward)
        self.E_backward = tf.Variable(e_backward)
        self.R_forward = tf.Variable(r_forward)
        self.R_backward = tf.Variable(r_backward)

        self.E_message_b = tf.Variable(bias_init)
        self.E_gate_b = tf.Variable(bias_init)
        self.R_message_b = tf.Variable(bias_init)
        self.R_gate_b = tf.Variable(bias_init)




    def local_get_weights(self):
        return [self.E_forward, self.E_message_b,
                self.E_backward, self.E_gate_b,
                self.R_forward, self.R_message_b,
                self.R_backward, self.R_gate_b]

    def dot_or_lookup(self, features, weights):
        if self.onehot_input:
            return tf.nn.embedding_lookup(weights, features)
        else:
            return tf.matmul(features, weights)

    def get_vertex_features(self, mode='train'):
        e_sender_index_vector = self.graph_representation.get_entity_sender_indices()
        e_receiver_index_vector = self.graph_representation.get_entity_receiver_indices()
        r_sender_index_vector = self.graph_representation.get_relation_sender_indices()
        r_receiver_index_vector = self.graph_representation.get_relation_receiver_indices()

        if self.onehot_input:
            return e_sender_index_vector, e_receiver_index_vector, r_sender_index_vector, r_receiver_index_vector
        else:
            e_code, r_code = self.next_component.compute_bipartite_embeddings(mode=mode)
            e_code = tf.nn.dropout(e_code, self.dropout_keep_probability)
            r_code = tf.nn.dropout(r_code, self.dropout_keep_probability)

            e_sender_codes = tf.nn.embedding_lookup(e_code, e_sender_index_vector)
            e_receiver_codes = tf.nn.embedding_lookup(e_code, e_receiver_index_vector)
            r_sender_codes = tf.nn.embedding_lookup(r_code, r_sender_index_vector)
            r_receiver_codes = tf.nn.embedding_lookup(r_code, r_receiver_index_vector)

            return e_sender_codes, e_receiver_codes, r_sender_codes, r_receiver_codes

    def get_all_codes(self, mode='train'):
        collected_messages = self.compute_bipartite_embeddings(mode=mode)[0]

        return collected_messages, None, collected_messages

    def compute_bipartite_embeddings(self, mode='train'):
        if self.vertex_embedding_function[mode] is None:
            features = self.get_vertex_features(mode=mode)
            messages = self.compute_messages(features)
            self.vertex_embedding_function[mode] = self.collect_messages(messages)

        return self.vertex_embedding_function[mode]

    def compute_messages(self, features):
        e_forward_messages = tf.nn.relu(self.dot_or_lookup(features[0], self.E_forward) + self.E_forward_b)
        e_backward_messages = tf.nn.relu(self.dot_or_lookup(features[1], self.E_backward) + self.E_backward_b)
        r_forward_messages = tf.nn.relu(self.dot_or_lookup(features[2], self.R_forward) + self.R_forward_b)
        r_backward_messages = tf.nn.relu(self.dot_or_lookup(features[3], self.R_backward) + self.R_backward_b)

        return e_forward_messages, e_backward_messages, r_forward_messages, r_backward_messages

    def collect_messages(self, messages):
        e_forward_mtr = self.graph_representation.get_entity_forward_v_by_m(normalized=True)
        e_backward_mtr = self.graph_representation.get_entity_backward_v_by_m(normalized=True)
        r_forward_mtr = self.graph_representation.get_relation_forward_v_by_m(normalized=True)
        r_backward_mtr = self.graph_representation.get_relation_backward_v_by_m(normalized=True)

        collected_e_messages = tf.sparse_tensor_dense_matmul(r_forward_mtr, messages[2])
        collected_e_messages += tf.sparse_tensor_dense_matmul(r_backward_mtr, messages[3])
        collected_r_messages = tf.sparse_tensor_dense_matmul(e_forward_mtr, messages[0])
        collected_r_messages += tf.sparse_tensor_dense_matmul(e_backward_mtr, messages[1])

        return collected_e_messages, collected_r_messages

    def get_all_subject_codes(self, mode='train'):
        return self.compute_vertex_embeddings(mode=mode)

    def get_all_object_codes(self, mode='train'):
        return self.compute_vertex_embeddings(mode=mode)