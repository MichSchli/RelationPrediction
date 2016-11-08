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
        self.embedding_width = int(self.settings['EmbeddingWidth'])
        self.regularization_parameter = float(self.settings['RegularizationParameter'])
        #self.n_convolutions = int(self.settings['NumberOfConvolutions'])    

    def preprocess(self, triplets):
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

        scaled_message_to_receiver_matrix = degree_matrix * message_to_receiver_matrix
        rows, cols, vals = sps.find(scaled_message_to_receiver_matrix)

        #Create TF message-to-receiver matrix:
        self.MTR = tf.SparseTensor(np.array([rows,cols]).transpose(), vals.astype(np.float32), [self.entity_count, receiver_indices.shape[0]])

        #Create TF sender-to-message matrix:
        self.STM = tf.Variable(sender_indices, dtype=np.int32)

        #Create TF message type list:
        self.R = tf.Variable(message_types, dtype=np.int32)
        
    def initialize_test(self):
        self.X = tf.placeholder(tf.int32, shape=[None,3])

    def initialize_train(self):
        embedding_initial = np.random.randn(self.entity_count, self.embedding_width).astype(np.float32)
        type_initial = np.random.randn(self.relation_count*2+1, self.embedding_width).astype(np.float32)

        layer_initial_v = np.random.randn(self.embedding_width, self.embedding_width).astype(np.float32)
        layer_initial_t = np.random.randn(self.embedding_width, self.embedding_width).astype(np.float32)
        
        relation_initial = np.random.randn(self.relation_count, self.embedding_width).astype(np.float32)

        self.X = tf.placeholder(tf.int32, shape=[None,3])

        self.W_embedding = tf.Variable(embedding_initial)
        self.W_type = tf.Variable(type_initial)
        self.W_layer_v = tf.Variable(layer_initial_v)
        self.W_layer_t = tf.Variable(layer_initial_t)
        
        self.W_relation = tf.Variable(relation_initial)
        
    def get_vertex_embedding(self, training=False):
        # Get relation and vertex embeddings:
        T = tf.nn.embedding_lookup(self.W_type, self.R)
        M = tf.nn.embedding_lookup(self.W_embedding, self.STM)

        # Transform
        message_embed = tf.matmul(T, self.W_layer_t)
        message_embed += tf.matmul(M, self.W_layer_v)

        # Average:
        mean_message = tf.sparse_tensor_dense_matmul(self.MTR, message_embed)

        self.vertex_embedding = mean_message
        
    def get_all_subject_codes(self):
        if self.vertex_embedding is None:
            self.get_vertex_embedding()
            
        return self.vertex_embedding
    
    def get_all_object_codes(self):
        if self.vertex_embedding is None:
            self.get_vertex_embedding()
            
        return self.vertex_embedding
    
    def get_weights(self):
        return [self.W_embedding, self.W_relation, self.W_type, self.W_layer_t, self.W_layer_v]

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

    #Hack
    def parameter_count(self):
        return 5

    def assign_weights(self, weights):
        self.W_embedding = tf.Variable(weights[0])
        self.W_relation = tf.Variable(weights[1])
        self.W_relation = tf.Variable(weights[2])
        self.W_layer_v = tf.Variable(weights[3])
        self.W_layer_t = tf.Variable(weights[4])
