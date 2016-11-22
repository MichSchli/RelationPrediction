import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sps
import scipy
import imp

abstract = imp.load_source('abstract_encoder', 'code/experts/encoders/abstract_encoder.py')


class Encoder(abstract.Encoder):

    settings = None
    vertex_embedding = None
    
    def __init__(self, encoder_settings):
        self.settings = encoder_settings

        self.entity_count = int(self.settings['EntityCount'])
        self.relation_count = int(self.settings['RelationCount'])
        self.embedding_width = int(self.settings['EmbeddingWidth'])
        self.n_convolutions = int(self.settings['NumberOfConvolutions'])
        
    def preprocess(self, triplets):
        triplets = np.array(triplets).transpose()
        relations = triplets[1]

        # Arg1-matrix (A1 e = r)
        arg1_rows = np.arange(relations.shape[0])
        arg1_columns = triplets[0]

        #Arg2-matrix (A2 r = e)
        arg2_rows = triplets[2]
        arg2_columns = np.arange(relations.shape[0])

        values = np.ones_like(arg1_columns, dtype=np.int32)

        #Precompute normalization for speed:
        arg1_matrix = coo_matrix((values, (arg1_rows, arg1_columns)), shape=(arg1_columns.shape[0], self.entity_count), dtype=np.float32).tocsr()
        arg2_matrix = coo_matrix((values, (arg2_rows, arg2_columns)), shape=(self.entity_count, arg1_columns.shape[0]), dtype=np.float32).tocsr()

        self.J_r = tf.constant(1 / 3, dtype=np.float32)
        j_e = 1 / (arg1_matrix.sum(axis=0) + arg2_matrix.sum(axis=1).transpose() + 1)
        self.J_e = tf.constant(j_e.transpose(), dtype=np.float32)
        
        #Create TF versions of matrices:
        self.Arg1 = tf.SparseTensor(np.array([arg1_rows,arg1_columns]).transpose(), values.astype(np.float32), [arg1_columns.shape[0], self.entity_count])
        self.Arg1_t = tf.SparseTensor(np.array([arg1_columns, arg1_rows]).transpose(), values.astype(np.float32), [self.entity_count, arg1_columns.shape[0]])
        
        self.Arg2 = tf.SparseTensor(np.array([arg2_rows,arg2_columns]).transpose(), values.astype(np.float32), [self.entity_count, arg2_columns.shape[0]])
        self.Arg2_t = tf.SparseTensor(np.array([arg2_columns, arg2_rows]).transpose(), values.astype(np.float32), [arg2_columns.shape[0], self.entity_count])
        self.R = tf.constant(relations, dtype=np.int32)


    
    def initialize_test(self):
        self.X = tf.placeholder(tf.int32, shape=[None,3])

    def initialize_train(self):
        entity_embedding_initial = np.sqrt(2.0 / 1) * np.random.randn(self.entity_count, self.embedding_width).astype(np.float32)
        relation_embedding_initial = np.sqrt(2.0 / 1) * np.random.randn(self.relation_count, self.embedding_width).astype(np.float32)

        type_initials = [np.sqrt(2.0 / self.embedding_width) * np.random.randn(6, self.embedding_width, self.embedding_width).astype(np.float32)
                                for _ in range(self.n_convolutions)]
        
        relation_initial = np.random.randn(self.relation_count, self.embedding_width).astype(np.float32)

        self.X = tf.placeholder(tf.int32, shape=[None,3])

        self.W_entity_embedding = tf.Variable(entity_embedding_initial)
        self.W_relation_embedding = tf.Variable(relation_embedding_initial)
        self.W_types = [tf.Variable(init) for init in type_initials]
        self.W_relation = tf.Variable(relation_initial)
        
    def get_vertex_embedding(self, training=False):
        #Initialize with single dense layer:
        vertex_embedding = self.W_entity_embedding
        relation_embedding = tf.nn.embedding_lookup(self.W_relation_embedding, self.R)
        activated_vertex_embedding = vertex_embedding
        activated_relation_embedding = relation_embedding

        for W_type in self.W_types:
            # Create messages:
            transformed_es_arg1 = tf.matmul(activated_vertex_embedding, W_type[0])
            transformed_es_arg2 = tf.matmul(activated_vertex_embedding, W_type[1])
            transformed_rs_arg1 = tf.matmul(activated_relation_embedding, W_type[2])
            transformed_rs_arg2 = tf.matmul(activated_relation_embedding, W_type[3])

            # Propagate to relation-nodes:
            mean_relation_message = tf.matmul(activated_relation_embedding, W_type[4])
            mean_relation_message += tf.sparse_tensor_dense_matmul(self.Arg1, transformed_es_arg1)
            mean_relation_message += tf.sparse_tensor_dense_matmul(self.Arg2_t, transformed_es_arg2)
            mean_relation_message *= self.J_r
            
            # Propagate to entity-nodes:
            mean_entity_message = tf.matmul(activated_vertex_embedding, W_type[5])
            mean_entity_message += tf.sparse_tensor_dense_matmul(self.Arg2, transformed_rs_arg2)
            mean_entity_message += tf.sparse_tensor_dense_matmul(self.Arg1_t, transformed_rs_arg1)
            mean_entity_message *= self.J_e

            #Construct new embeddings:
            vertex_embedding = mean_entity_message
            activated_vertex_embedding = tf.nn.relu(vertex_embedding)
            
            relation_embedding = mean_relation_message
            activated_relation_embedding = tf.nn.relu(relation_embedding)

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
        return [self.W_entity_embedding, self.W_relation_embedding, self.W_relation] + self.W_types

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
        return 2 + self.n_convolutions

    def assign_weights(self, weights):
        self.W_entity_embedding = tf.Variable(weights[0])
        self.W_relation_embedding = tf.Variable(weights[1])
        self.W_relation = tf.Variable(weights[2])
        self.W_types = [tf.Variable(w) for w in weights[3:]]
