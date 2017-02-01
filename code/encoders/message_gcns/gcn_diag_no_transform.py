import numpy as np
import tensorflow as tf

from encoders.message_gcns.message_gcn import MessageGcn


class DiagGcnNoTransform(MessageGcn):

    def parse_settings(self):
        self.embedding_width = int(self.settings['InternalEncoderDimension'])
        self.dropout_keep_probability = float(self.settings['DropoutKeepProbability'])

    def local_initialize_train(self):
        type_matrix_shape = (self.relation_count *2 + 1, self.embedding_width)
        vertex_matrix_shape = (self.embedding_width, self.embedding_width)

        type_init_var = 1
        type_initial_f = np.random.normal(0, type_init_var, size=type_matrix_shape).astype(np.float32)
        type_initial_b = np.random.normal(0, type_init_var, size=type_matrix_shape).astype(np.float32)

        glorot_var_combined = np.sqrt(3 / (vertex_matrix_shape[0] + vertex_matrix_shape[1]))

        self_initial = np.random.normal(0, glorot_var_combined, size=vertex_matrix_shape).astype(np.float32)

        message_bias = np.zeros(self.embedding_width).astype(np.float32)

        self.V_types_f = tf.Variable(type_initial_f)
        self.V_types_b = tf.Variable(type_initial_b)
        self.V_self = tf.Variable(self_initial)
        self.B_message = tf.Variable(message_bias)

    def local_get_weights(self):
        return [self.V_types_f, self.V_types_b, self.V_self,
                self.B_message]

    def compute_messages(self, sender_features, receiver_features):
        message_types = self.get_graph().get_type_indices()
        type_diags_f = tf.nn.embedding_lookup(self.V_types_f, message_types)
        type_diags_b = tf.nn.embedding_lookup(self.V_types_b, message_types)

        terms_f = tf.mul(sender_features, type_diags_f)
        terms_b = tf.mul(sender_features, type_diags_b)

        return terms_f, terms_b

    def compute_self_loop_messages(self, vertex_features):
        return self.dot_or_lookup(vertex_features, self.V_self)

    def combine_messages(self, forward_messages, backward_messages, self_loop_messages, previous_code, mode='train'):
        mtr_f = self.get_graph().forward_incidence_matrix(normalization=('global', 'recalculated'))
        mtr_b = self.get_graph().backward_incidence_matrix(normalization=('global', 'recalculated'))

        collected_messages_f = tf.sparse_tensor_dense_matmul(mtr_f, forward_messages)
        collected_messages_b = tf.sparse_tensor_dense_matmul(mtr_b, backward_messages)

        new_embedding = self_loop_messages + collected_messages_f + collected_messages_b + self.B_message

        if self.use_nonlinearity:
            new_embedding = tf.nn.relu(new_embedding)

        return new_embedding