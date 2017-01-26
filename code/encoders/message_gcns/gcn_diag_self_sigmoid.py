import numpy as np
import tensorflow as tf

from encoders.message_gcns.message_gcn import MessageGcn


class DiagGcn(MessageGcn):

    def parse_settings(self):
        self.embedding_width = int(self.settings['InternalEncoderDimension'])
        self.dropout_keep_probability = float(self.settings['DropoutKeepProbability'])

    def local_initialize_train(self):
        vertex_feature_dimension = self.entity_count if self.onehot_input else self.embedding_width
        type_matrix_shape = (self.relation_count * 2, self.embedding_width)
        vertex_matrix_shape = (vertex_feature_dimension, self.embedding_width)

        glorot_var_combined = np.sqrt(3/(vertex_matrix_shape[0]*2 + vertex_matrix_shape[1]))
        type_init_var = 1

        self_initial = np.random.normal(0, glorot_var_combined, size=vertex_matrix_shape).astype(np.float32)
        vertex_projection_sender_initial = np.random.normal(0, glorot_var_combined, size=vertex_matrix_shape).astype(np.float32)

        glorot_self_gate = np.sqrt(3/(vertex_matrix_shape[0] + vertex_matrix_shape[1]))
        glorot_message_gate = np.sqrt(3 /(2*self.embedding_width))

        self_gate_initial = np.random.normal(0, glorot_self_gate, size=vertex_matrix_shape).astype(np.float32)
        self_message_initial = np.random.normal(0, glorot_message_gate, size=(self.embedding_width, self.embedding_width)).astype(np.float32)

        type_initial = np.random.normal(0, type_init_var, size=type_matrix_shape).astype(np.float32)

        message_bias = np.zeros(self.embedding_width).astype(np.float32)

        self.V_proj_sender = tf.Variable(vertex_projection_sender_initial)
        self.V_self = tf.Variable(self_initial)
        self.V_types = tf.Variable(type_initial)

        self.G_self = tf.Variable(self_gate_initial)
        self.G_message = tf.Variable(self_message_initial)

        self.B_message = tf.Variable(message_bias)
        self.B_self = tf.Variable(message_bias)
        self.B_gate = tf.Variable(message_bias)


    def local_get_weights(self):
        return [self.V_proj_sender, self.V_types, self.V_self,
                self.G_self, self.G_message,
                self.B_message, self.B_self, self.B_gate]

    def compute_messages(self, sender_features):
        sender_terms = self.dot_or_lookup(sender_features, self.V_proj_sender)
        message_types = self.get_graph().get_type_indices()
        type_diags = tf.nn.embedding_lookup(self.V_types, message_types)

        terms = tf.mul(sender_terms, type_diags)

        messages = terms
        return messages

    def compute_self_loop_messages(self, vertex_features):
        return self.dot_or_lookup(vertex_features, self.V_self)


    def combine_messages(self, messages, self_loop_messages, vertex_features):
        mtr = self.get_graph().incidence_matrix(normalization=('global', 'recalculated'))
        collected_messages = tf.sparse_tensor_dense_matmul(mtr, messages)

        gates = tf.nn.sigmoid(self.dot_or_lookup(vertex_features, self.G_self) + tf.matmul(collected_messages, self.G_message) + self.B_gate)

        if self.use_nonlinearity:
            self_loop_messages = tf.nn.relu(self_loop_messages + self.B_self)
            collected_messages = tf.nn.relu(collected_messages + self.B_message)


        new_embedding = tf.mul(gates, self_loop_messages) + tf.mul(1-gates, collected_messages)

        return new_embedding
