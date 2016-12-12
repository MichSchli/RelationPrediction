import numpy as np
import theano
from theano import tensor as T
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sps
import scipy
import imp

abstract = imp.load_source('abstract_encoder', 'code/experts/encoders/abstract_message_based_graph_encoder.py')

class Encoder(abstract.TheanoEncoder):

    settings = None
    vertex_embedding = None
    
    def __init__(self, encoder_settings):
        abstract.TheanoEncoder.__init__(self, encoder_settings)
        self.settings = encoder_settings

        self.graph_width = int(self.settings['GraphWidth'])
        self.embedding_width = int(self.settings['EmbeddingWidth'])
        self.n_convolutions = int(self.settings['NumberOfConvolutions'])
        self.regularization_parameter = float(self.settings['RegularizationParameter'])
        self.dropout_probability = float(self.settings['DropoutProbability'])


    
    def initialize_test(self):
        self.X = T.matrix('X', dtype='int32')
    
    def initialize_train(self):
        embedding_initial = np.sqrt(2.0) * np.random.randn(self.entity_count, self.graph_width).astype(np.float32)        
        relation_initial = np.sqrt(2.0) * np.random.randn(self.relation_count, self.embedding_width).astype(np.float32)
        type_initials = [np.sqrt(2.0 / self.embedding_width) *
                         np.random.randn(self.relation_count*2+1, self.graph_width, self.graph_width).astype(np.float32)
                         for _ in range(self.n_convolutions)]

        projection_initial = np.sqrt(2.0) / self.embedding_width * np.random.randn(self.graph_width, self.embedding_width).astype(np.float32)        

        self.X = T.matrix('X', dtype='int32')

        self.W_embedding = theano.shared(embedding_initial)
        self.W_relation = theano.shared(relation_initial)
        self.W_projection = theano.shared(projection_initial)
        self.W_types = [theano.shared(t) for t in type_initials]

    def apply_dropouts(self, layer, dropout_probability):
        if self.srng is None:
            self.srng = RandomStreams(seed=12345)

        keep_probability = 1 - dropout_probability

        mask = self.srng.binomial(n=1, p=keep_probability, size=layer.shape, dtype='float32')
        output = layer * T.cast(mask, theano.config.floatX) / keep_probability
        return output
        
    def get_vertex_embedding(self, training=False):
        activated_embedding = self.W_embedding

        for W_type in self.W_types:
            # Apply layer-wise dropout:
            if training:
                activated_embedding = self.apply_dropouts(activated_embedding, self.dropout_probability)

            mean_messages = T.zeros_like(activated_embedding)
            for e_to_r, e_to_v, v_to_e in zip(self.E_to_R, self.E_to_V, self.V_to_E):
                #Fetch 
                MessageTransforms = W_type[e_to_r]

                #Gather values from vertices in message matrix:
                MessageValues = activated_embedding[e_to_v]

                #Transform according to 
                Messages = T.batched_dot(MessageTransforms, MessageValues)
                mean_messages += theano.sparse.dot(v_to_e, Messages)

            vertex_embedding = mean_messages
            activated_embedding = T.nnet.relu(mean_messages)

        #No activation for final layer:
        if training:
                activated_embedding = self.apply_dropouts(activated_embedding, self.dropout_probability)
                
        self.vertex_embedding = T.dot(activated_embedding, self.W_projection)
        
    def get_all_subject_codes(self):
        if self.vertex_embedding is None:
            self.get_vertex_embedding()
            
        return self.vertex_embedding
    
    def get_all_object_codes(self):
        if self.vertex_embedding is None:
            self.get_vertex_embedding()

        return self.vertex_embedding
    
    def get_weights(self):
        return [self.W_embedding, self.W_relation, self.W_projection] + self.W_types

    def get_input_variables(self):
        return [self.X]

    def encode(self, training=True):
        if self.vertex_embedding is None:
            self.get_vertex_embedding()
            
        self.e1s = self.vertex_embedding[self.X[:,0]]
        self.rs = self.W_relation[self.X[:,1]]
        self.e2s = self.vertex_embedding[self.X[:,2]]

        return self.e1s, self.rs, self.e2s

    def get_regularization(self):
        regularization = T.sqr(self.e1s).mean() + T.sqr(self.rs).mean() + T.sqr(self.e2s).mean()
        return self.regularization_parameter * regularization

    def parameter_count(self):
        return 3 + self.n_convolutions
    
    def assign_weights(self, weights):
        self.W_embedding = theano.shared(weights[0])
        self.W_relation = theano.shared(weights[1])
        self.W_projection = theano.shared(weights[2])
        self.W_types = [theano.shared(w) for w in weights[3:]]
