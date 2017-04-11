import numpy as np
from scipy.sparse import coo_matrix
import math
import tensorflow as tf
from model import Model


class MessageGraph():

    sender_indices = None
    receiver_indices = None
    message_types = None

    def __init__(self, edges, vertex_count, label_count):
        self.vertex_count = vertex_count
        self.label_count = label_count
        self.edges = edges

        self.process(self.edges)

    def process(self, triplets):
        triplets = tf.transpose(triplets)
        self.sender_indices = triplets[0]
        self.receiver_indices = triplets[2]
        self.message_types = triplets[1]

        self.edge_count = tf.shape(self.sender_indices)[0]


    def get_sender_indices(self):
        return self.sender_indices

    def get_type_indices(self):
        return self.message_types

    def get_receiver_indices(self):
        return self.receiver_indices

    '''
    def compute_normalized_values(self, receiver_indices, message_types):
        if self.normalization[0] == "global":
            mrs = receiver_indices
        else:
            mrs = [tuple(x) for x in np.vstack((receiver_indices, message_types)).transpose()]

        counts = {}
        for mr in mrs:
            if mr in counts:
                counts[mr] += 1.0
            else:
                counts[mr] = 1.0

        return np.array([1.0 / counts[mr] for mr in mrs]).astype(np.float32)

    def compute_sparse_mtr(self):
        if self.normalization[0] != "none":
            mtr_values = self.compute_normalized_mtr(self.receiver_indices, self.message_types)
        else:
            mtr_values = np.ones_like(self.message_types).astype(np.int32)

        message_indices = np.arange(self.edge_count).astype(np.int32)

        mtr_indices = np.vstack((self.receiver_indices, message_indices)).transpose()
        mtr_shape = [self.vertex_count, self.edge_count]

        return mtr_indices, mtr_values, mtr_shape
    '''

    def forward_incidence_matrix(self, normalization):
        if normalization[0] == "none":
            mtr_values = tf.to_float(tf.ones_like(self.receiver_indices))
            message_indices = tf.range(self.edge_count)

            mtr_indices = tf.to_int64(tf.transpose(tf.stack([self.receiver_indices, message_indices])))
            mtr_shape = tf.to_int64(tf.stack([self.vertex_count, self.edge_count]))

            tensor = tf.SparseTensor(indices=mtr_indices,
                                     values=mtr_values,
                                     dense_shape=mtr_shape)

            return tensor
        elif normalization[0] == "global":
            mtr_values = tf.to_float(tf.ones_like(self.receiver_indices))
            message_indices = tf.range(self.edge_count)

            mtr_indices = tf.to_int64(tf.transpose(tf.stack([self.receiver_indices, message_indices])))
            mtr_shape = tf.to_int64(tf.stack([self.vertex_count, self.edge_count]))

            tensor = tf.sparse_softmax(tf.SparseTensor(indices=mtr_indices,
                                   values=mtr_values,
                                                       dense_shape=mtr_shape))

            return tensor
        elif normalization[0] == "local":
            mtr_values = tf.to_float(tf.ones_like(self.receiver_indices))
            message_indices = tf.range(self.edge_count)

            mtr_indices = tf.to_int64(tf.transpose(tf.stack([self.message_types, self.receiver_indices, message_indices])))
            mtr_shape = tf.to_int64(tf.stack([self.label_count*2, self.vertex_count, self.edge_count]))

            tensor = tf.sparse_softmax(tf.SparseTensor(indices=mtr_indices,
                                   values=mtr_values,
                                                       dense_shape=mtr_shape))

            tensor = tf.sparse_reduce_sum_sparse(tensor, 0)

            return tensor

    def backward_incidence_matrix(self, normalization):
        if normalization[0] == "none":
            mtr_values = tf.to_float(tf.ones_like(self.sender_indices))
            message_indices = tf.range(self.edge_count)

            mtr_indices = tf.to_int64(tf.transpose(tf.stack([self.sender_indices, message_indices])))
            mtr_shape = tf.to_int64(tf.stack([self.vertex_count, self.edge_count]))

            tensor = tf.SparseTensor(indices=mtr_indices,
                                     values=mtr_values,
                                     dense_shape=mtr_shape)

            return tensor
        elif normalization[0] == "global":
            mtr_values = tf.to_float(tf.ones_like(self.sender_indices))
            message_indices = tf.range(self.edge_count)

            mtr_indices = tf.to_int64(tf.transpose(tf.stack([self.sender_indices, message_indices])))
            mtr_shape = tf.to_int64(tf.stack([self.vertex_count, self.edge_count]))

            tensor = tf.sparse_softmax(tf.SparseTensor(indices=mtr_indices,
                                   values=mtr_values,
                                                       dense_shape=mtr_shape))

            return tensor
        elif normalization[0] == "local":
            mtr_values = tf.to_float(tf.ones_like(self.sender_indices))
            message_indices = tf.range(self.edge_count)

            mtr_indices = tf.to_int64(tf.transpose(tf.stack([self.message_types, self.sender_indices, message_indices])))
            mtr_shape = tf.to_int64(tf.stack([self.label_count*2, self.vertex_count, self.edge_count]))

            tensor = tf.sparse_softmax(tf.SparseTensor(indices=mtr_indices,
                                   values=mtr_values,
                                   dense_shape=mtr_shape))

            tensor = tf.sparse_reduce_sum_sparse(tensor, 0)

            return tensor


class Representation(Model):

    normalization="global"
    graph = None
    X = None

    def __init__(self, triples, settings, bipartite=False):
        self.triples = np.array(triples)
        self.entity_count = settings['EntityCount']
        self.relation_count = settings['RelationCount']
        self.edge_count = self.triples.shape[0]*2

        #self.process(self.triples)

        #self.graph = None#MessageGraph(triples, self.entity_count, self.relation_count)


    def get_graph(self):
        if self.graph is None:
            self.graph = MessageGraph(self.X, self.entity_count, self.relation_count)

        return self.graph

    def local_initialize_train(self):
        self.X = tf.placeholder(tf.int32, shape=[None, 3], name='graph_edges')

    def local_get_train_input_variables(self):
        return [self.X]

    def local_get_test_input_variables(self):
        return [self.X]

    '''
    def compute_normalized_values(self, receiver_indices, message_types):
        if self.normalization == "global":
            mrs = receiver_indices
        else:
            mrs = [tuple(x) for x in np.vstack((receiver_indices, message_types)).transpose()]

        counts = {}
        for mr in mrs:
            if mr in counts:
                counts[mr] += 1.0
            else:
                counts[mr] = 1.0

        return np.array([1.0 / counts[mr] for mr in mrs]).astype(np.float32)

    def compute_sparse_mtr(self):
        if self.normalization != "none":
            mtr_values = self.compute_normalized_values(self.receiver_indices, self.message_types)
        else:
            mtr_values = np.ones_like(self.message_types).astype(np.int32)

        message_indices = np.arange(self.edge_count).astype(np.int32)

        mtr_indices = np.vstack((self.receiver_indices, message_indices)).transpose()
        mtr_shape = [self.entity_count, self.edge_count]

        return mtr_indices, mtr_values, mtr_shape

    def process(self, triplets):
        triplets = triplets.transpose()
        self.sender_indices = np.hstack((triplets[0], triplets[2])).astype(np.int32)
        self.receiver_indices = np.hstack((triplets[2], triplets[0])).astype(np.int32)
        self.message_types = np.hstack(
            (triplets[1], triplets[1] + self.relation_count)).astype(np.int32)
    '''
