import imp
import numpy as np
import tensorflow as tf
import theano
from theano import tensor as T
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sps
import scipy
from theano import sparse

abstract = imp.load_source('abstract_encoder', 'code/experts/encoders/abstract_encoder.py')

def __preprocess__(triplets):
    triplets = np.array(triplets).transpose()
    relations = triplets[1]

    sender_indices = np.hstack((triplets[0], triplets[2], np.arange(self.entity_count))).astype(np.int32)
    receiver_indices = np.hstack((triplets[2], triplets[0], np.arange(self.entity_count))).astype(np.int32)
    message_types = np.hstack((triplets[1]+1, triplets[1]+self.relation_count+1, np.zeros(self.entity_count))).astype(np.int32)

    message_indices = np.arange(receiver_indices.shape[0], dtype=np.int32)
    values = np.ones_like(receiver_indices, dtype=np.int32)

    message_to_receiver_matrix = coo_matrix((values, (receiver_indices, message_indices)), shape=(self.entity_count, receiver_indices.shape[0]), dtype=np.float32).tocsr()

    degrees = (1 / message_to_receiver_matrix.sum(axis=1)).tolist()
    degree_matrix = sps.lil_matrix((self.entity_count, self.entity_count))
    degree_matrix.setdiag(degrees)

    scaled_vertex_to_edge_sparse = degree_matrix * message_to_receiver_matrix
    edge_to_vertex_list = receiver_indices
    edge_to_relation_list = message_types
    
    return scaled_vertex_to_edge_sparse, edge_to_vertex_list, edge_to_relation_list

class TheanoEncoder(abstract.Encoder):
    def __init__(self):
        pass

    def preprocess(self, triplets):
        scaled_vertex_to_edge_sparse, edge_to_vertex_list, edge_to_relation_list = __preprocess__(triplets)

        self.V_by_E = scaled_vertex_to_edge_sparse
        self.E_by_V = T.TensorConstant(edge_to_vertex_list)
        self.E_by_R = T.TensorConstant(edge_to_relation_list)

        

class TensorflowEncoder(abstract.Encoder):

    def __init__(self):
        pass

    def preprocess(self, triplets):
        scaled_vertex_to_edge_sparse, edge_to_vertex_list, edge_to_relation_list = __preprocess__(triplets)
        rows, cols, vals = sps.find(scaled_vertex_to_edge_sparse)

        #Create TF message-to-receiver matrix:
        self.MTR = tf.SparseTensor(np.array([rows,cols]).transpose(), vals.astype(np.float32), [self.entity_count, receiver_indices.shape[0]])

        #Create TF sender-to-message matrix:
        self.STM = tf.constant(edge_to_vertex_list, dtype=np.int32)

        #Create TF message type list:
        self.R = tf.constant(edge_to_relation_list, dtype=np.int32)
