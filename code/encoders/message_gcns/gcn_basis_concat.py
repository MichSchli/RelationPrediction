import numpy as np
import tensorflow as tf
from common.shared_functions import dot_or_lookup, glorot_variance, make_tf_variable, make_tf_bias

from encoders.message_gcns.message_gcn import MessageGcn


class ConcatGcn(MessageGcn):

    def parse_settings(self):
        self.dropout_keep_probability = float(self.settings['DropoutKeepProbability'])

        self.n_coefficients = int(self.settings['NumberOfBasisFunctions'])

        self.submatrix_d = int(self.shape[1]/self.n_coefficients)

    def local_initialize_train(self):
        vertex_feature_dimension = self.entity_count if self.onehot_input else self.submatrix_d
        vertex_matrix_shape = (self.relation_count, self.n_coefficients, vertex_feature_dimension, self.submatrix_d)
        self_matrix_shape = self.shape

        glorot_var_combined = glorot_variance([vertex_matrix_shape[0], vertex_matrix_shape[2]])
        self.W_forward = make_tf_variable(0, glorot_var_combined, vertex_matrix_shape)
        self.W_backward = make_tf_variable(0, glorot_var_combined, vertex_matrix_shape)
        self.W_self = make_tf_variable(0, glorot_var_combined, self_matrix_shape)

        self.b = make_tf_bias(self.shape[1])


    def local_get_weights(self):
        return [self.W_forward, self.W_backward,
                self.W_self,
                self.b]

    def compute_messages(self, sender_features, receiver_features):
        # Compute total list of transforms:
        message_types = self.get_graph().get_type_indices()
        forward_transforms = tf.nn.embedding_lookup(self.W_forward, message_types)
        backward_transforms = tf.nn.embedding_lookup(self.W_backward, message_types)

        # Reshape vertex embeddings:
        reshaped_s_f = tf.reshape(sender_features, [-1, self.n_coefficients, self.submatrix_d])
        reshaped_r_f = tf.reshape(receiver_features, [-1, self.n_coefficients, self.submatrix_d])

        # Transform embeddings:
        forward_messages = tf.squeeze(tf.matmul(forward_transforms, tf.expand_dims(reshaped_s_f, -1)))
        backward_messages = tf.squeeze(tf.matmul(backward_transforms, tf.expand_dims(reshaped_r_f, -1)))

        # Return:
        forward_messages_rshp = tf.reshape(forward_messages, [-1, self.shape[1]])
        backward_messages_rshp = tf.reshape(backward_messages, [-1, self.shape[1]])
        return forward_messages_rshp, backward_messages_rshp


    def dot_or_tensor_mul(self, features, tensor):
        tensor_shape = tf.shape(tensor)
        flat_shape = [tensor_shape[0], tensor_shape[1] * tensor_shape[2]]

        flattened_tensor = tf.reshape(tensor, flat_shape)
        result_tensor = dot_or_lookup(features, flattened_tensor, onehot_input=self.onehot_input)
        result_tensor = tf.reshape(result_tensor, [-1, tensor_shape[1], tensor_shape[2]])

        return result_tensor

    def compute_self_loop_messages(self, vertex_features):
        return dot_or_lookup(vertex_features, self.W_self, onehot_input=self.onehot_input)


    def combine_messages(self, forward_messages, backward_messages, self_loop_messages, previous_code, mode='train'):
        mtr_f = self.get_graph().forward_incidence_matrix(normalization=('global', 'recalculated'))
        mtr_b = self.get_graph().backward_incidence_matrix(normalization=('global', 'recalculated'))

        collected_messages_f = tf.sparse_tensor_dense_matmul(mtr_f, forward_messages)
        collected_messages_b = tf.sparse_tensor_dense_matmul(mtr_b, backward_messages)

        updated_vertex_embeddings = collected_messages_f + collected_messages_b

        if self.use_nonlinearity:
            activated = tf.nn.relu(updated_vertex_embeddings + self_loop_messages)
        else:
            activated = updated_vertex_embeddings + self_loop_messages

        return activated

    def local_get_regularization(self):
        regularization = tf.reduce_mean(tf.square(self.W_forward))
        regularization += tf.reduce_mean(tf.square(self.W_backward))
        regularization += tf.reduce_mean(tf.square(self.W_self))

        return 0.0 * regularization