import imp
import numpy as np
import tensorflow as tf
import theano
from theano import tensor as T
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sps
import scipy
from theano import sparse
import math

abstract = imp.load_source('abstract_encoder', 'code/experts/encoders/abstract_encoder.py')

class AMBGE(abstract.Encoder):

    graph_batch_size = 1000000
    n_slices = None
    
    def __init__(self, settings):
        abstract.Encoder.__init__(self, settings)
        self.use_global_normalization = settings['UseGlobalNorm'] == 'True'

    def compute_normalized_values(self, receiver_indices, message_types):
        if self.use_global_normalization:
            mrs = receiver_indices
        else:
            mrs = [tuple(x) for x in np.vstack((receiver_indices, message_types)).transpose()]

        counts = {}
        for mr in mrs:
            if mr in counts:
                counts[mr] += 1.0
            else:
                counts[mr] = 1.0

        return np.array([1.0/counts[mr] for mr in mrs]).astype(np.float32)

    def compute_sparse_mtrs(self, receiver_indices, message_types):
        normalized_values = self.compute_normalized_values(receiver_indices, message_types)
        
        v = np.array_split(normalized_values, self.n_slices)
        i = np.array_split(receiver_indices, self.n_slices)

        mtrs = [None] * self.n_slices
        for idx in range(self.n_slices):
            j = np.arange(v[idx].shape[0], dtype=np.int32)
            mtrs[idx] = coo_matrix((v[idx], (i[idx], j)), shape=(self.entity_count, v[idx].shape[0]), dtype=np.float32).tocsr()
        
        return mtrs
    
    def __preprocess__(self, triplets):
        triplets = np.array(triplets).transpose()
        relations = triplets[1]

        sender_indices = np.hstack((triplets[0], triplets[2], np.arange(self.entity_count))).astype(np.int32)
        receiver_indices = np.hstack((triplets[2], triplets[0], np.arange(self.entity_count))).astype(np.int32)
        message_types = np.hstack((triplets[1]+1, triplets[1]+self.relation_count+1, np.zeros(self.entity_count))).astype(np.int32)

        self.n_slices = math.ceil((len(sender_indices) / float(self.graph_batch_size)))

        sparse_mtrs = self.compute_sparse_mtrs(receiver_indices, message_types)
        edge_to_vertex_lists = np.array_split(receiver_indices, self.n_slices)
        edge_to_relation_lists = np.array_split(message_types, self.n_slices)
    
        return sparse_mtrs, edge_to_vertex_lists, edge_to_relation_lists

class TheanoEncoder(AMBGE):

    def preprocess(self, triplets):
        scaled_vertex_to_edge_sparse, edge_to_vertex_list, edge_to_relation_list = self.__preprocess__(triplets)

        print(scaled_vertex_to_edge_sparse)
        self.V_to_E = scaled_vertex_to_edge_sparse
        self.E_to_V = edge_to_vertex_list
        self.E_to_R = edge_to_relation_list
        

class TensorflowEncoder(AMBGE):

    '''
    TODO: BROKEN
    '''

    def preprocess(self, triplets):
        scaled_vertex_to_edge_sparse, edge_to_vertex_list, edge_to_relation_list = self.__preprocess__(triplets)
        rows, cols, vals = sps.find(scaled_vertex_to_edge_sparse)

        #Create TF message-to-receiver matrix:
        self.MTR = tf.SparseTensor(np.array([rows,cols]).transpose(), vals.astype(np.float32), [self.entity_count, edge_to_vertex_list.shape[0]])

        #Create TF sender-to-message matrix:
        self.STM = tf.constant(edge_to_vertex_list, dtype=np.int32)

        #Create TF message type list:
        self.R = tf.constant(edge_to_relation_list, dtype=np.int32)
