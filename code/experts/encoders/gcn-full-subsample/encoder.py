import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sps
import scipy
import imp

abstract = imp.load_source('abstract_encoder', 'code/experts/encoders/abstract_message_based_graph_encoder.py')

class Encoder(abstract.Encoder):

    settings = None
    vertex_embedding = None
    
    def __init__(self, encoder_settings):
        self.settings = encoder_settings

        self.entity_count = int(self.settings['EntityCount'])
        self.relation_count = int(self.settings['RelationCount'])
        self.edge_count = int(self.settings['EdgeCount'])
        self.embedding_width = int(self.settings['EmbeddingWidth'])
        self.regularization_parameter = float(self.settings['RegularizationParameter'])
        self.message_dropout_probability = float(self.settings['MessageDropoutProbability'])
        #self.n_convolutions = int(self.settings['NumberOfConvolutions'])
        self.subsample_size = int(self.settings['SubsampleSize'])
    
        
    def initialize_test(self):
        self.X = tf.placeholder(tf.int32, shape=[None,3])

    def initialize_train(self):
        embedding_initial = np.random.randn(self.entity_count, self.embedding_width).astype(np.float32)
        type_initials = np.random.randn(self.relation_count*2+1, self.embedding_width, self.embedding_width).astype(np.float32)
        relation_initial = np.random.randn(self.relation_count, self.embedding_width).astype(np.float32)

        self.X = tf.placeholder(tf.int32, shape=[None,3])

        self.W_embedding = tf.Variable(embedding_initial)
        self.W_types = tf.Variable(type_initials)
        self.W_relation = tf.Variable(relation_initial)

        '''
        Subsampling
        '''
        self.edge_subset = tf.random_uniform([self.subsample_size], 0, self.edge_count*2+self.entity_count, tf.int32)
        
        #Initialize messages to random values:
        self.previous_messages = tf.Variable(np.random.randn(self.edge_count*2+self.entity_count, self.embedding_width).astype(np.float32))

        
    def get_vertex_embedding(self, training=False):
        activated_embedding = self.W_embedding

        self.edge_subset = tf.random_crop(tf.range(self.edge_count*2+self.entity_count), [self.subsample_size])
        
        #Get subset of edges:
        relations = tf.nn.embedding_lookup(self.R, self.edge_subset)
        senders = tf.nn.embedding_lookup(self.STM, self.edge_subset)
        prevs = tf.nn.embedding_lookup(self.previous_messages, self.edge_subset)
        
        T = tf.nn.embedding_lookup(self.W_types, relations)
        M = tf.nn.embedding_lookup(activated_embedding, senders)
        M_prime = tf.squeeze(tf.batch_matmul(tf.expand_dims(M,1), T))
        M_change = M_prime - prevs

        columns = tf.range(self.subsample_size)
        indices = tf.concat(1, [tf.expand_dims(self.edge_subset,1), tf.expand_dims(columns, 1)])
        values = tf.ones(self.subsample_size)        

        # Perform update:
        assignment_matrix = tf.SparseTensor(tf.to_int64(indices), values, [self.edge_count*2+self.entity_count, self.subsample_size])
        assignment = tf.sparse_tensor_dense_matmul(assignment_matrix, M_change)
        updated_messages = self.previous_messages + assignment

        mean_message_per_vertex = tf.sparse_tensor_dense_matmul(self.MTR, updated_messages)

        self.message_update_operation = self.previous_messages.assign(updated_messages)
        self.vertex_embedding = mean_message_per_vertex

    def get_additional_ops(self):
        return [self.message_update_operation]
        
    def get_all_subject_codes(self):
        if self.vertex_embedding is None:
            self.get_vertex_embedding()
            
        return self.vertex_embedding
    
    def get_all_object_codes(self):
        if self.vertex_embedding is None:
            self.get_vertex_embedding()

        return self.vertex_embedding
    
    def get_weights(self):
        return [self.W_embedding, self.W_relation, self.W_types]

    def get_input_variables(self):
        return [self.X]

    def encode(self, training=True):
        if self.vertex_embedding is None:
            self.get_vertex_embedding()
            
        self.e1s = tf.nn.embedding_lookup(self.vertex_embedding, self.X[:,0])
        self.rs = tf.nn.embedding_lookup(self.W_relation, self.X[:,1])
        self.e2s = tf.nn.embedding_lookup(self.vertex_embedding, self.X[:,2])

        return self.e1s, self.rs, self.e2s

    def get_regularization(self):
        #regularization = tf.reduce_mean(tf.square(self.e1s))
        #regularization += tf.reduce_mean(tf.square(self.rs))
        #regularization += tf.reduce_mean(tf.square(self.e2s))

        return 0 #self.regularization_parameter * regularization

    def parameter_count(self):
        return 3

    def assign_weights(self, weights):
        self.W_embedding = tf.Variable(weights[0])
        self.W_relation = tf.Variable(weights[1])
        self.W_types = tf.Variable(weights[2])
