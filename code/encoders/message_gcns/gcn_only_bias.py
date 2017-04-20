import numpy as np
import tensorflow as tf
from common.shared_functions import dot_or_lookup, glorot_variance, make_tf_variable, make_tf_bias

from encoders.message_gcns.message_gcn import MessageGcn


class OnlyBiasGcn(MessageGcn):

    def parse_settings(self):
        self.dropout_keep_probability = float(self.settings['DropoutKeepProbability'])
        self.graph_batch_size = int(self.settings['GraphBatchSize'])


    def local_initialize_train(self):
        bias_matrix_shape = (self.relation_count, self.shape[1])

        glorot_var_combined = glorot_variance(bias_matrix_shape)
        self.b_forward = make_tf_variable(0, glorot_var_combined, bias_matrix_shape)
        self.b_backward = make_tf_variable(0, glorot_var_combined, bias_matrix_shape)


    def local_get_weights(self):
        return [self.b_forward,
                self.b_backward]

    def compute_messages(self, sender_features, receiver_features):
        message_types = self.get_graph().get_type_indices()

        forward_messages = tf.nn.embedding_lookup(self.b_forward, message_types)
        backward_messages = tf.nn.embedding_lookup(self.b_backward, message_types)
        return forward_messages, backward_messages

    def compute_self_loop_messages(self, vertex_features):
        return None


    def combine_messages(self, forward_messages, backward_messages, self_loop_messages, previous_code, mode='train'):
        mtr_f = self.get_graph().forward_incidence_matrix(normalization=('global', 'recalculated'))
        mtr_b = self.get_graph().backward_incidence_matrix(normalization=('global', 'recalculated'))

        collected_messages_f = tf.sparse_tensor_dense_matmul(mtr_f, forward_messages)
        collected_messages_b = tf.sparse_tensor_dense_matmul(mtr_b, backward_messages)

        updated_vertex_embeddings = collected_messages_f + collected_messages_b

        if self.use_nonlinearity:
            activated = tf.nn.relu(updated_vertex_embeddings)
        else:
            activated = updated_vertex_embeddings

        return activated

    def local_get_regularization(self):
        regularization = tf.reduce_mean(tf.square(self.b_forward))
        regularization += tf.reduce_mean(tf.square(self.b_backward))

        return 0.0 * regularization