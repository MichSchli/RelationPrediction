import numpy as np
import tensorflow as tf
from common.shared_functions import dot_or_lookup, glorot_variance, make_tf_variable, make_tf_bias

from encoders.message_gcns.message_gcn import MessageGcn


class BasisGcnStore(MessageGcn):

    def parse_settings(self):
        self.dropout_keep_probability = float(self.settings['DropoutKeepProbability'])
        self.graph_batch_size = int(self.settings['GraphBatchSize'])

        self.n_coefficients = int(self.settings['NumberOfBasisFunctions'])

    def local_initialize_train(self):
        vertex_feature_dimension = self.entity_count if self.onehot_input else self.shape[0]
        type_matrix_shape = (self.relation_count, self.n_coefficients)
        vertex_matrix_shape = (vertex_feature_dimension, self.n_coefficients, self.shape[1])
        self_matrix_shape = (vertex_feature_dimension, self.shape[1])

        glorot_var_combined = glorot_variance([vertex_matrix_shape[0], vertex_matrix_shape[2]])
        self.W_forward = make_tf_variable(0, glorot_var_combined, vertex_matrix_shape)
        self.W_backward = make_tf_variable(0, glorot_var_combined, vertex_matrix_shape)
        self.W_self = make_tf_variable(0, glorot_var_combined, self_matrix_shape)

        type_init_var = 1
        self.C_forward = make_tf_variable(0, type_init_var, type_matrix_shape)
        self.C_backward = make_tf_variable(0, type_init_var, type_matrix_shape)

        self.b = make_tf_bias(self.shape[1])

        self.cached_vertex_embeddings = tf.Variable(np.zeros(self_matrix_shape, dtype=np.float32))
        self.cached_messages_f = tf.Variable(np.zeros((self.edge_count, self.shape[1]), dtype=np.float32))
        self.cached_messages_b = tf.Variable(np.zeros((self.edge_count, self.shape[1]), dtype=np.float32))

        self.I = tf.placeholder(tf.int32, shape=[None], name="batch_indices")

    def local_get_train_input_variables(self):
        return [self.I]

    def local_get_weights(self):
        return [self.W_forward, self.W_backward,
                self.C_forward, self.C_backward,
                self.W_self,
                self.b]

    def compute_messages(self, sender_features, receiver_features):
        backward_type_scaling, forward_type_scaling = self.compute_coefficients()
        receiver_terms, sender_terms = self.compute_basis_functions(receiver_features, sender_features)

        forward_messages = tf.reduce_sum(sender_terms * tf.expand_dims(forward_type_scaling,-1), 1)
        backward_messages = tf.reduce_sum(receiver_terms * tf.expand_dims(backward_type_scaling, -1), 1)

        return forward_messages, backward_messages

    def compute_coefficients(self):
        message_types = self.get_graph().get_type_indices()
        forward_type_scaling = tf.nn.embedding_lookup(self.C_forward, message_types)
        backward_type_scaling = tf.nn.embedding_lookup(self.C_backward, message_types)
        return backward_type_scaling, forward_type_scaling

    def compute_basis_functions(self, receiver_features, sender_features):
        sender_terms = self.dot_or_tensor_mul(sender_features, self.W_forward)
        receiver_terms = self.dot_or_tensor_mul(receiver_features, self.W_backward)

        return receiver_terms, sender_terms

    def dot_or_tensor_mul(self, features, tensor):
        tensor_shape = tf.shape(tensor)
        flat_shape = [tensor_shape[0], tensor_shape[1] * tensor_shape[2]]

        flattened_tensor = tf.reshape(tensor, flat_shape)
        result_tensor = dot_or_lookup(features, flattened_tensor, onehot_input=self.onehot_input)
        result_tensor = tf.reshape(result_tensor, [-1, tensor_shape[1], tensor_shape[2]])

        return result_tensor

    def compute_self_loop_messages(self, vertex_features):
        return dot_or_lookup(vertex_features, self.W_self, onehot_input=self.onehot_input)


    def get_additional_ops(self):
        return [self.f_upd, self.b_upd, self.v_upd]

    def combine_messages(self, forward_messages, backward_messages, self_loop_messages, previous_code, mode='train'):
        mtr_f = self.get_graph().forward_incidence_matrix(normalization=('none', 'recalculated'))
        mtr_b = self.get_graph().backward_incidence_matrix(normalization=('none', 'recalculated'))

        if mode == 'train':
            forward_messages_comp = forward_messages - tf.nn.embedding_lookup(self.cached_messages_f, self.I)
            backward_messages_comp = backward_messages - tf.nn.embedding_lookup(self.cached_messages_b, self.I)

            with tf.control_dependencies([forward_messages, backward_messages]):
                self.f_upd = tf.scatter_update(self.cached_messages_f, self.I, forward_messages)
                self.b_upd = tf.scatter_update(self.cached_messages_b, self.I, backward_messages)

            collected_messages_f = tf.sparse_tensor_dense_matmul(mtr_f, forward_messages_comp)
            collected_messages_b = tf.sparse_tensor_dense_matmul(mtr_b, backward_messages_comp)

            new_embedding = collected_messages_f + collected_messages_b
            updated_vertex_embeddings = new_embedding + self.cached_vertex_embeddings

            with tf.control_dependencies([updated_vertex_embeddings]):
                self.v_upd = tf.assign(self.cached_vertex_embeddings, updated_vertex_embeddings)
        else:
            collected_messages_f = tf.sparse_tensor_dense_matmul(mtr_f, forward_messages)
            collected_messages_b = tf.sparse_tensor_dense_matmul(mtr_b, backward_messages)

            new_embedding = collected_messages_f + collected_messages_b
            updated_vertex_embeddings = new_embedding


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