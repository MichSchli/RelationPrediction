import numpy as np
import tensorflow as tf

class Encoder():

    settings = None
    
    def __init__(self, encoder_settings):
        self.settings = encoder_settings

        self.entity_count = int(self.settings['EntityCount'])
        self.relation_count = int(self.settings['RelationCount'])
        self.embedding_width = int(self.settings['EmbeddingWidth'])
        self.regularization_parameter = float(self.settings['RegularizationParameter'])

    def initialize_test(self):
        self.X = tf.placeholder(tf.int32, shape=[None,3])

    def preprocess(self, triplets):
        pass
    
    def initialize_train(self):
        embedding_initial = np.random.randn(self.entity_count, self.embedding_width).astype(np.float32)        
        relation_initial = np.random.randn(self.relation_count, self.embedding_width).astype(np.float32)

        self.X = tf.placeholder(tf.int32, shape=[None,3])

        self.W_embedding = tf.Variable(embedding_initial)
        self.W_relation = tf.Variable(relation_initial)

    def get_all_subject_codes(self):
        return self.W_embedding
    
    def get_all_object_codes(self):
        return self.W_embedding
    
    def get_weights(self):
        return [self.W_embedding, self.W_relation]

    def get_input_variables(self):
        return [self.X]

    def encode(self, training=True):
        self.e1s = tf.nn.embedding_lookup(self.W_embedding, self.X[:,0])
        self.rs = tf.nn.embedding_lookup(self.W_relation, self.X[:,1])
        self.e2s = tf.nn.embedding_lookup(self.W_embedding, self.X[:,2])

        return self.e1s, self.rs, self.e2s

    def get_regularization(self):
        regularization = tf.reduce_mean(tf.square(self.e1s))
        regularization += tf.reduce_mean(tf.square(self.rs))
        regularization += tf.reduce_mean(tf.square(self.e2s))

        return self.regularization_parameter * regularization

    def parameter_count(self):
        return 2

    def assign_weights(self, weights):
        self.W_embedding = tf.Variable(weights[0])
        self.W_relation = tf.Variable(weights[1])
