import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sps
import scipy
import imp

abstract = imp.load_source('abstract_encoder', 'code/experts/encoders/abstract_message_based_graph_encoder.py')

class Encoder(abstract.TensorflowEncoder):

    settings = None
    vertex_embedding = None
    
    def __init__(self, encoder_settings):
        self.settings = encoder_settings

        self.entity_count = int(self.settings['EntityCount'])
        self.relation_count = int(self.settings['RelationCount'])
        self.embedding_width = int(self.settings['EmbeddingWidth'])
        self.regularization_parameter = float(self.settings['RegularizationParameter'])
        self.message_dropout_probability = float(self.settings['MessageDropoutProbability'])
        self.n_convolutions = int(self.settings['NumberOfConvolutions'])    
    
        
    def initialize_test(self):
        self.X = tf.placeholder(tf.int32, shape=[None,3])

    def initialize_train(self):
        embedding_initial = np.random.randn(self.entity_count, self.embedding_width).astype(np.float32)
        type_initials = [np.random.randn(self.relation_count*2+1, self.embedding_width).astype(np.float32)
                                for _ in range(self.n_convolutions)]
        
        relation_initial = np.random.randn(self.relation_count, self.embedding_width).astype(np.float32)

        self.X = tf.placeholder(tf.int32, shape=[None,3])

        self.W_embedding = tf.Variable(embedding_initial)
        self.W_types = [tf.Variable(init) for init in type_initials]
        self.W_relation = tf.Variable(relation_initial)
        
    def get_vertex_embedding(self, training=False):
        vertex_embedding = self.W_embedding

        #No activation for first layer. Maybe subject to change.
        activated_embedding = vertex_embedding

        for W_type in self.W_types:
            T = tf.nn.embedding_lookup(W_type, self.R)

            #Gather values from vertices in message matrix:
            M = tf.nn.embedding_lookup(activated_embedding, self.STM)

            #Transform messages according to types:
            M_prime = tf.squeeze(tf.mul(M, T))
                
            #Construct new vertex embeddings:
            mean_message = tf.sparse_tensor_dense_matmul(self.MTR, M_prime)
            vertex_embedding = mean_message

            if training:
                activated_embedding = tf.nn.dropout(vertex_embedding, self.message_dropout_probability)

            activated_embedding = tf.nn.relu(vertex_embedding)

        #No activation for final layer:
        self.vertex_embedding = vertex_embedding
        
    def get_all_subject_codes(self):
        if self.vertex_embedding is None:
            self.get_vertex_embedding()
            
        return self.vertex_embedding
    
    def get_all_object_codes(self):
        if self.vertex_embedding is None:
            self.get_vertex_embedding()

        return self.vertex_embedding
    
    def get_weights(self):
        return [self.W_embedding, self.W_relation] + self.W_types

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
        regularization = tf.reduce_mean(tf.square(self.e1s))
        regularization += tf.reduce_mean(tf.square(self.rs))
        regularization += tf.reduce_mean(tf.square(self.e2s))

        return self.regularization_parameter * regularization

    def parameter_count(self):
        return 2 + self.n_convolutions

    def assign_weights(self, weights):
        self.W_embedding = tf.Variable(weights[0])
        self.W_relation = tf.Variable(weights[1])
        self.W_types = [tf.Variable(w) for w in weights[2:]]
