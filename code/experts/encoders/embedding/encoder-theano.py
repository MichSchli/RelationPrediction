import numpy as np
import theano
from theano import tensor as T
import imp

abstract = imp.load_source('abstract_encoder', 'code/experts/encoders/abstract_encoder.py')

class Encoder(abstract.Encoder):

    settings = None
    
    def __init__(self, encoder_settings):
        self.settings = encoder_settings

        self.entity_count = int(self.settings['EntityCount'])
        self.relation_count = int(self.settings['RelationCount'])
        self.embedding_width = int(self.settings['EmbeddingWidth'])
        self.regularization_parameter = float(self.settings['RegularizationParameter'])


    def initialize_test(self):
        self.X = T.matrix('X', dtype='int32')
    
    def initialize_train(self):
        embedding_initial = np.random.randn(self.entity_count, self.embedding_width).astype(np.float32)        
        relation_initial = np.random.randn(self.relation_count, self.embedding_width).astype(np.float32)

        self.X = T.matrix('X', dtype='int32')

        self.W_embedding = theano.shared(embedding_initial)
        self.W_relation = theano.shared(relation_initial)
        
    def get_all_subject_codes(self):
        return self.W_embedding
    
    def get_all_object_codes(self):
        return self.W_embedding
    
    def get_weights(self):
        return [self.W_embedding, self.W_relation]

    def encode(self, training=True):
        self.e1s = self.W_embedding[self.X[:,0]]
        self.rs = self.W_relation[self.X[:,1]]
        self.e2s = self.W_embedding[self.X[:,2]]
        
        return self.e1s, self.rs, self.e2s

    def get_regularization(self):
        regularization = T.sqr(self.e1s).mean()
        regularization += T.sqr(self.rs).mean()
        regularization += T.sqr(self.e2s).mean()

        return self.regularization_parameter * regularization

    def parameter_count(self):
        return 2

    def assign_weights(self, weights):
        self.W_embedding = theano.shared(weights[0])
        self.W_relation = theano.shared(weights[1])
