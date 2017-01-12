import numpy as np
import tensorflow as tf
from model import Model


class SimpleGatedBasisGcn(Model):
    V_proj = None
    E_proj = None
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
        vertex_feature_dimension = self.entity_count if self.onehot_input else self.embedding_width

        vertex_matrix_shape = (vertex_feature_dimension, self.embedding_width)
        edge_matrix_shape = (self.relation_count*2+1, self.embedding_width)

        glorot_var_combined = np.sqrt(3/(vertex_matrix_shape[0] + vertex_matrix_shape[1]))

        vertex_projection_sender_initial = np.random.normal(0, glorot_var_combined, size=vertex_matrix_shape).astype(np.float32)

        vertex_gate_sender_initial = np.random.normal(0, glorot_var_combined, size=vertex_matrix_shape).astype(np.float32)
        edge_gate_initial = np.random.normal(0, glorot_var_combined, size=edge_matrix_shape).astype(np.float32)

        message_bias = np.zeros(self.embedding_width).astype(np.float32)
        gate_pre_bias = np.zeros(self.embedding_width).astype(np.float32)

        self.V_proj_sender = tf.Variable(vertex_projection_sender_initial)

        self.V_gate_sender = tf.Variable(vertex_gate_sender_initial)
        self.E_gate = tf.Variable(edge_gate_initial)

        self.B_message = tf.Variable(message_bias)
        self.B_gate_pre = tf.Variable(gate_pre_bias)

    def local_get_weights(self):
        return [self.V_proj_sender,
                self.V_gate_sender, self.E_gate,
                self.B_message, self.B_gate_pre]

    def dot_or_lookup(self, features, weights):
        if self.onehot_input:
            return tf.nn.embedding_lookup(weights, features)
        else:
            return tf.matmul(features, weights)

    def get_vertex_features(self, mode='train'):
        sender_index_vector = self.graph_representation.get_sender_indices()
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
            type_index_vector = self.graph_representation.get_type_indices()
            messages = self.compute_messages(sender_features)
            gates = self.compute_gate_values(sender_features, type_index_vector)
            gated_messages = tf.mul(messages, gates)
            self.vertex_embedding_function[mode] = self.collect_messages(gated_messages)

        return self.vertex_embedding_function[mode]

    def compute_gate_values(self, sender_features, type_index_vector):
        sender_gate_terms = self.dot_or_lookup(sender_features, self.V_gate_sender)
        type_terms = tf.nn.embedding_lookup(self.E_gate, type_index_vector)
        gate_energies = sender_gate_terms + type_terms + self.B_gate_pre

        flat_energies = tf.reshape(gate_energies, [-1])
        gates = self.apply_edge_softmax(flat_energies)
        return gates

    def compute_messages(self, sender_features):
        sender_terms = self.dot_or_lookup(sender_features, self.V_proj_sender)
        messages = tf.nn.relu(sender_terms + self.B_message)
        return messages

    def collect_messages(self, gated_messages):
        receiver_index_vector = self.graph_representation.get_receiver_indices()
        edge_count = receiver_index_vector.shape[0]

        mtr_indices = np.vstack((receiver_index_vector, np.arange(edge_count))).transpose()
        mtr_values = np.ones_like(receiver_index_vector).astype(np.float32)
        mtr_shape = [self.entity_count, edge_count]
        mtr = tf.SparseTensor(indices=mtr_indices,
                              values=mtr_values,
                              shape=mtr_shape)
        collected_messages = tf.sparse_tensor_dense_matmul(mtr, gated_messages)
        return collected_messages

    def apply_edge_softmax(self, flat_energies):
        receiver_index_vector = self.graph_representation.get_receiver_indices()
        edge_count = receiver_index_vector.shape[0]

        repeating_message_indices = np.repeat(np.arange(edge_count), self.embedding_width)
        repeating_receivers = np.repeat(receiver_index_vector, self.embedding_width)
        repeating_dim_indexes = np.tile(np.arange(self.embedding_width), edge_count)
        combined_indexes = np.transpose(
            np.vstack((repeating_receivers, repeating_dim_indexes, repeating_message_indices)))
        gate_energy_matrix = tf.SparseTensor(indices=combined_indexes,
                                             values=flat_energies,
                                             shape=[self.entity_count, self.embedding_width, edge_count])
        gate_energy_matrix_with_softmax = tf.sparse_softmax(gate_energy_matrix)
        gates = tf.transpose(tf.reshape(gate_energy_matrix_with_softmax.values, [self.embedding_width, -1]))
        return gates

    def get_all_subject_codes(self, mode='train'):
        return self.compute_vertex_embeddings(mode=mode)

    def get_all_object_codes(self, mode='train'):
        return self.compute_vertex_embeddings(mode=mode)