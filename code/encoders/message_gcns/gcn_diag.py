import numpy as np
import tensorflow as tf
from common.shared_functions import dot_or_lookup, glorot_variance, make_tf_variable, make_tf_bias

from encoders.message_gcns.message_gcn import MessageGcn


class DiagGcn(MessageGcn):

    def parse_settings(self):
        self.embedding_width = int(self.settings['InternalEncoderDimension'])
        self.dropout_keep_probability = float(self.settings['DropoutKeepProbability'])

    def local_initialize_train(self):
        type_matrix_shape = (self.relation_count, self.embedding_width)
        vertex_matrix_shape = (self.embedding_width, self.embedding_width)

        glorot_var_self = glorot_variance(vertex_matrix_shape)
        self.V_self = make_tf_variable(0, glorot_var_self, vertex_matrix_shape)

        type_init_var = 1
        self.V_types_f = make_tf_variable(0, type_init_var, type_matrix_shape)
        self.V_types_b = make_tf_variable(0, type_init_var, type_matrix_shape)

        self.B_message = make_tf_bias(self.embedding_width)

    def local_get_weights(self):
        return [self.V_types_f, self.V_types_b, self.V_self,
                self.B_message]

    def compute_messages(self, sender_features, receiver_features):
        message_types = self.get_graph().get_type_indices()
        type_diags_f = tf.nn.embedding_lookup(self.V_types_f, message_types)
        type_diags_b = tf.nn.embedding_lookup(self.V_types_b, message_types)

        terms_f = tf.mul(sender_features, type_diags_f)
        terms_b = tf.mul(receiver_features, type_diags_b)

        return terms_f, terms_b

    def compute_self_loop_messages(self, vertex_features):
        return dot_or_lookup(vertex_features, self.V_self, onehot_input=self.onehot_input)

    def combine_messages(self, forward_messages, backward_messages, self_loop_messages, previous_code, mode='train'):
        mtr_f = self.get_graph().forward_incidence_matrix(normalization=('global', 'recalculated'))
        mtr_b = self.get_graph().backward_incidence_matrix(normalization=('global', 'recalculated'))

        collected_messages_f = tf.sparse_tensor_dense_matmul(mtr_f, forward_messages)
        collected_messages_b = tf.sparse_tensor_dense_matmul(mtr_b, backward_messages)

        new_embedding = self_loop_messages + collected_messages_f + collected_messages_b + self.B_message

        if self.use_nonlinearity:
            new_embedding = tf.nn.relu(new_embedding)

        return new_embedding