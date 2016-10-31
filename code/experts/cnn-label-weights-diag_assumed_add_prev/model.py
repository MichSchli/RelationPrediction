import numpy as np
import pickle
import random
import imp
import tensorflow as tf
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sps
import scipy

abstract_model = imp.load_source('abstract_model', 'code/experts/AbstractModel.py')
shared = imp.load_source('shared', 'code/experts/shared.py')
evaluation = imp.load_source('evaluation', 'code/evaluation/evaluation.py')

class Model():

    '''
    Fields:
    '''
    
    model_path = None
    backend = "tensorflow"
    
    n_entities = None
    n_relations = None

    batch_size = 10000
    embedding_width = 200
    number_of_negative_samples = 10
    regularization_parameter = 0.01

    n_convolutions = 1
    
    positives_forward = None
    positives_backward = None

    message_dropout_probability = 0.5
    prev_dropout_probability = 0.5
    
    '''
    Initialization methods:
    '''
    
    def __init__(self):
        pass

    def set_model_path(self, model_path):
        self.model_path = model_path

    def preprocess(self, triplets):
        self.graph_edges = triplets

        triplets = np.array(triplets).transpose()
        
        relations = triplets[1]

        sender_indices = np.hstack((triplets[0], triplets[2])).astype(np.int32)
        receiver_indices = np.hstack((triplets[2], triplets[0])).astype(np.int32)
        message_types = np.hstack((triplets[1], triplets[1]+self.n_relations)).astype(np.int32)

        message_indices = np.arange(receiver_indices.shape[0], dtype=np.int32)
        values = np.ones_like(receiver_indices, dtype=np.int32)

        message_to_receiver_matrix = coo_matrix((values, (receiver_indices, message_indices)), shape=(self.n_entities, receiver_indices.shape[0]), dtype=np.float32).tocsr()

        degrees = (1 / message_to_receiver_matrix.sum(axis=1)).tolist()
        degree_matrix = sps.lil_matrix((self.n_entities, self.n_entities))
        degree_matrix.setdiag(degrees)

        scaled_message_to_receiver_matrix = degree_matrix * message_to_receiver_matrix
        rows, cols, vals = sps.find(scaled_message_to_receiver_matrix)

        #Create TF message-to-receiver matrix:
        self.MTR = tf.SparseTensor(np.array([rows,cols]).transpose(), vals.astype(np.float32), [self.n_entities, receiver_indices.shape[0]])

        #Create TF sender-to-message matrix:
        self.STM = tf.constant(sender_indices, dtype=np.int32)

        #Create TF message type list:
        self.R = tf.constant(message_types, dtype=np.int32)
        
            
    def set_entity_count(self, count):
        self.n_entities = count

    def set_relation_count(self, count):
        self.n_relations = count
    
    '''
    Negative sampling:
    '''

    def transform(self, triplets):
        return self.process_train_triplets(triplets, self.graph_edges)
    
    def process_train_triplets(self, triplet_sample, all_triplets, disable_saving=False):
        new_labels = np.zeros((len(triplet_sample) * (self.number_of_negative_samples + 1 ))).astype(np.float32)
        new_indexes = np.tile(triplet_sample, (self.number_of_negative_samples + 1,1)).astype(np.int32)
        new_labels[:len(triplet_sample)] = 1

        #if self.positives_forward is None:
        #    self.positives_forward, self.positives_backward = self.generate_positive_sample_dictionaries(all_triplets)

        number_to_generate = len(triplet_sample)*self.number_of_negative_samples
        choices = np.random.binomial(1, 0.5, number_to_generate)

        total = range(self.n_entities)

        for i in range(self.number_of_negative_samples):
            for j, triplet in enumerate(triplet_sample):
                index = i*len(triplet_sample)+j

                if choices[index]:
                    #positive_objects = self.positives_forward[triplet[0]][triplet[1]]

                    found = False
                    while not found:
                        sample = random.choice(total)
                        if True: #sample not in positive_objects:
                            new_indexes[index+len(triplet_sample),2] = sample
                            found = True
                else:
                    #positive_subjects = self.positives_backward[triplet[2]][triplet[1]]

                    found = False
                    while not found:
                        sample = random.choice(total)
                        if True: #sample not in positive_subjects:
                            new_indexes[index+len(triplet_sample),0] = sample
                            found = True

        if disable_saving:
            self.positives_forward = None
            self.positives_backward = None

        return new_indexes, new_labels

    def generate_positive_sample_dictionaries(self, triplets_in_kb):
        positives_forward = {}
        positives_backward = {}
        for triplet in triplets_in_kb:
            if triplet[0] not in positives_forward:
                positives_forward[triplet[0]] = {triplet[1] : [triplet[2]]}
            else:
                if triplet[1] not in positives_forward[triplet[0]]:
                    positives_forward[triplet[0]][triplet[1]] = [triplet[2]]
                else:
                    positives_forward[triplet[0]][triplet[1]].append(triplet[2])

            if triplet[2] not in positives_backward:
                positives_backward[triplet[2]] = {triplet[1] : [triplet[0]]}
            else:
                if triplet[1] not in positives_backward[triplet[2]]:
                    positives_backward[triplet[2]][triplet[1]] = [triplet[0]]
                else:
                    positives_backward[triplet[2]][triplet[1]].append(triplet[0])

        return positives_forward, positives_backward
    
    def expand_triplets(self, triplets):
        triplet_array = np.array(triplets).astype(np.int32)
        organized = np.transpose(triplet_array)
        return organized[0], organized[2], organized[1]

    '''
    Training:
    '''

    def get_optimizer_input_variables(self):
        return [self.X, self.Y]
    
    def get_optimizer_parameters(self):
        return [('Minibatches', {'batch_size':self.batch_size, 'contiguous_sampling':False}),
                ('SampleTransformer', {'transform_function': self.transform}),
                ('IterationCounter', {'max_iterations':50000}),
                ('GradientClipping', {'max_norm':1}),
                ('Adam', {'learning_rate':0.01,
                          'historical_moment_weight':0.9,
                          'historical_gradient_weight':0.999}),
                ('TrainLossReporter', {'evaluate_every_n':5}),
                ('EarlyStopper', {'criteria':'score_validation_data',
                                  'evaluate_every_n':100,
                                  'scoring_function':self.score_validation_data,
                                  'comparator':lambda current, prev: current > prev,
                                  'burnin':20000}),
                ('ModelSaver', {'save_function': self.save,
                                'model_path': self.model_path,
                                'save_every_n':100})]
    
    def initialize_variables(self):
        embedding_initial = np.random.randn(self.n_entities, self.embedding_width).astype(np.float32)
        type_initial = np.random.randn(self.n_relations*2+1, self.embedding_width).astype(np.float32)

        convolution_initials_l = [np.random.randn(self.embedding_width, self.embedding_width).astype(np.float32)
                                for _ in range(self.n_convolutions)]
        convolution_initials_p = [np.random.randn(self.embedding_width, self.embedding_width).astype(np.float32)
                                for _ in range(self.n_convolutions)]

        
        relation_initial = np.random.randn(self.n_relations, self.embedding_width).astype(np.float32)

        self.X = tf.placeholder(tf.int32, shape=[None,3])
        self.Y = tf.placeholder(tf.float32, shape=[None])

        self.W_embedding = tf.Variable(embedding_initial)
        self.W_type = tf.Variable(type_initial)
        self.W_convolutions_l = [tf.Variable(init) for init in convolution_initials_l]
        self.W_convolutions_p = [tf.Variable(init) for init in convolution_initials_p]
        
        self.W_relation = tf.Variable(relation_initial)

    def get_vertex_embedding(self, training=False):
        vertex_embedding = self.W_embedding
        type_embedding = self.W_type

        #No activation for first layer. Maybe subject to change.
        activated_embedding = vertex_embedding

        T = tf.nn.embedding_lookup(type_embedding, self.R)
        for W_layer_l, W_layer_p in zip(self.W_convolutions_l, self.W_convolutions_p):

            #Gather values from vertices in message matrix:
            M = tf.nn.embedding_lookup(activated_embedding, self.STM)

            #Transform messages according to types:
            M_prime = tf.squeeze(tf.mul(M, T))
                
            if training:
                M_prime = tf.nn.dropout(M_prime, self.message_dropout_probability)

            #Construct new vertex embeddings:
            mean_message = tf.sparse_tensor_dense_matmul(self.MTR, M_prime)

            vertex_embedding = tf.matmul(mean_message, W_layer_l)

            if training:
                activated_embedding = tf.nn.dropout(activated_embedding, self.prev_dropout_probability)
                
            vertex_embedding += tf.matmul(activated_embedding, W_layer_p)
            
            activated_embedding = tf.nn.relu(vertex_embedding)

        #No activation for final layer:
        return vertex_embedding
        
    def get_optimizer_loss(self):
        vertex_embedding = self.get_vertex_embedding(training=True)

        regularization = tf.reduce_mean(tf.square(self.W_embedding)) + tf.reduce_mean(tf.square(self.W_type))
        for W_layer_l, W_layer_p in zip(self.W_convolutions_l, self.W_convolutions_p):
            regularization += tf.reduce_mean(tf.square(W_layer_l))
            regularization += tf.reduce_mean(tf.square(W_layer_p))
            
        e1s = tf.nn.embedding_lookup(vertex_embedding, self.X[:,0])
        rs = tf.nn.embedding_lookup(self.W_relation, self.X[:,1])
        e2s = tf.nn.embedding_lookup(vertex_embedding, self.X[:,2])

        regularization += tf.reduce_mean(tf.square(rs))
        
        energies = tf.reduce_sum(e1s*rs*e2s, 1)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(energies, self.Y))
        
        return loss + self.regularization_parameter * regularization

    def get_prediction(self, X):
        vertex_embedding = self.W_embedding
        type_embedding = self.W_type

        #No activation for first layer. Maybe subject to change.
        activated_embedding = vertex_embedding
        
        T = tf.nn.embedding_lookup(type_embedding, self.R)
        for W_layer in self.W_convolutions:
            #Gather values from vertices in message matrix:
            M = tf.nn.embedding_lookup(activated_embedding, self.STM)

            #Transform messages according to types:
            M_prime = tf.squeeze(tf.mul(M, T))
            M_prime = tf.nn.tanh(M_prime)

            #Construct new vertex embeddings:
            mean_message = tf.sparse_tensor_dense_matmul(self.MTR, M_prime)
            vertex_embedding = tf.matmul(mean_message, W_layer)
            activated_embedding = vertex_embedding #tf.nn.tanh(vertex_embedding)

        e1s = tf.nn.embedding_lookup(vertex_embedding, X[:,0])
        rs = tf.nn.embedding_lookup(self.W_relation, X[:,1])
        e2s = tf.nn.embedding_lookup(vertex_embedding, X[:,2])
        
        energies = tf.reduce_sum(e1s*rs*e2s, 1)
        return tf.nn.sigmoid(energies)
    
    def get_optimizer_weights(self):
        return [self.W_embedding, self.W_relation, self.W_type] + self.W_convolutions_l + self.W_convolutions_p

    #Fast, ugly, eval:

    
    def score_validation_data(self, validation_data):
        scorer = evaluation.Scorer()
        scorer.register_data(self.graph_edges)
        scorer.register_data(validation_data)
        scorer.register_model(self)

        score_summary = scorer.compute_scores(validation_data, verbose=True).get_summary()
        return score_summary.results['Filtered'][score_summary.mrr_string()]
    
    def initiate_eval(self):
        self.X = tf.placeholder(tf.int32, shape=[None,3])
        
        init_op = tf.initialize_all_variables()

        self.session.run(init_op)

    def compute_o(self):
        vertex_embedding = self.get_vertex_embedding()
        
        e1s = tf.nn.embedding_lookup(vertex_embedding, self.X[:,0])
        rs = tf.nn.embedding_lookup(self.W_relation, self.X[:,1])
        
        thingy2 = tf.matmul(e1s*rs, tf.transpose(vertex_embedding))
        return tf.nn.sigmoid(thingy2)
        
    def compute_s(self):
        vertex_embedding = self.get_vertex_embedding()

        rs = tf.nn.embedding_lookup(self.W_relation, self.X[:,1])
        e2s = tf.nn.embedding_lookup(vertex_embedding, self.X[:,2])
        
        thingy1 = tf.transpose(tf.matmul(vertex_embedding, tf.transpose(rs*e2s)))
        return tf.nn.sigmoid(thingy1)
    
    def score_all_subjects(self, tup):
        return self.session.run(self.compute_s(), feed_dict={self.X:tup})

    def score_all_objects(self, tup):
        return self.session.run(self.compute_o(), feed_dict={self.X:tup})

    #####

    def predict(self, triples):
        sess = tf.Session()

        X = tf.placeholder(tf.int32, shape=[len(triples), 3])
        init_op = tf.initialize_all_variables()

        sess.run(init_op)

        return sess.run(self.get_prediction(X), feed_dict={X:triples})
    
    '''
    To be replaced by inherited methods:
    '''
    
    def save(self, filename):
        store_package = self.session.run(self.get_optimizer_weights())

        store_package += [self.n_entities,
                         self.n_relations]

        store_file = open(filename, 'wb')
        pickle.dump(store_package, store_file)
        store_file.close()

    def load(self, filename):
        store_file = open(filename, 'rb')
        store_package = pickle.load(store_file)

        self.W_embedding = tf.Variable(store_package[0])
        self.W_relation = tf.Variable(store_package[1])
        self.W_type = tf.Variable(store_package[2])
        self.W_convolutions_l = [tf.Variable(store_package[x]) for x in range(3,3+self.n_convolutions)]
        self.W_convolutions_p = [tf.Variable(store_package[x]) for x in range(3+self.n_convolutions,-2)]
                               
        self.n_entities = store_package[-2]
        self.n_relations = store_package[-1]

        self.model_path = filename
    
