import numpy as np
import pickle
import random
import imp
import tensorflow as tf

abstract_model = imp.load_source('abstract_model', 'code/experts/AbstractModel.py')
shared = imp.load_source('shared', 'code/experts/shared.py')

class Model():

    '''
    Fields:
    '''
    
    model_path = None
    backend = "tensorflow"
    
    n_entities = None
    n_relations = None

    batch_size = 1000 #4831
    embedding_width = 200
    number_of_negative_samples = 10
    regularization_parameter = 0.01

    positives_forward = None
    positives_backward = None
    
    '''
    Initialization methods:
    '''
    
    def __init__(self):
        pass

    def set_model_path(self, model_path):
        self.model_path = model_path

    def preprocess(self, triplets):
        self.graph_edges = triplets
    
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
                #('GradientDescent', {'learning_rate':1.0}),
                #('AdaGrad', {'learning_rate':0.5}),
                ('Adam', {'learning_rate':0.001, 'historical_moment_weight':0.9, 'historical_gradient_weight':0.999}),
                ('EarlyStopper', {'criteria':'loss', 'evaluate_every_n':500}),
                ('ModelSaver', {'save_function': self.save, 'model_path': self.model_path})]
    
    def initialize_variables(self):
        embedding_initial = np.random.randn(self.n_entities, self.embedding_width).astype(np.float32)
        relation_initial = np.random.randn(self.n_relations, self.embedding_width).astype(np.float32)
        hidden1_initial = np.random.randn(500, self.embedding_width*3).astype(np.float32)
        hidden2_initial = np.random.randn(1,500).astype(np.float32)
        
        self.X = tf.placeholder(tf.int32, shape=[None,3])
        self.Y = tf.placeholder(tf.float32, shape=[None])

        self.W_embedding = tf.Variable(embedding_initial)
        self.W_relation = tf.Variable(relation_initial)
        self.H1 = tf.Variable(hidden1_initial)
        self.H2 = tf.Variable(hidden2_initial)

    def get_optimizer_loss(self):
        e1s = tf.nn.embedding_lookup(self.W_embedding, self.X[:,0])
        rs = tf.nn.embedding_lookup(self.W_relation, self.X[:,1])
        e2s = tf.nn.embedding_lookup(self.W_embedding, self.X[:,2])

        total = tf.concat(1, [e1s, rs, e2s])
        h1 = tf.nn.tanh(tf.matmul(self.H1, tf.transpose(total)))
        h2 = tf.matmul(self.H2, h1)
        
        energies = tf.transpose(tf.squeeze(h2))#tf.reduce_sum(e1s*rs*e2s, 1)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(energies, self.Y))
        
        regularization = tf.reduce_mean(tf.square(e1s)) + tf.reduce_mean(tf.square(rs)) + tf.reduce_mean(tf.square(e2s)) + tf.reduce_mean(tf.square(self.H1)) + tf.reduce_mean(tf.square(self.H2))
        
        return loss + self.regularization_parameter * regularization

    def get_prediction(self, X):
        e1s = tf.nn.embedding_lookup(self.W_embedding, X[:,0])
        rs = tf.nn.embedding_lookup(self.W_relation, X[:,1])
        e2s = tf.nn.embedding_lookup(self.W_embedding, X[:,2])

        total = tf.concat(1, [e1s, rs, e2s])
        h1 = tf.nn.tanh(tf.matmul(self.H1, tf.transpose(total)))
        h2 = tf.matmul(self.H2, h1)
        
        energies = tf.transpose(tf.squeeze(h2))
        return tf.nn.sigmoid(energies)
    
    def get_optimizer_weights(self):
        return [self.W_embedding, self.W_relation, self.H1, self.H2]


    def predict(self, triples):
        sess = tf.Session()

        X = tf.placeholder(tf.int32, shape=[len(triples), 3])
        init_op = tf.initialize_all_variables()

        sess.run(init_op)

        return sess.run(self.get_prediction(X), feed_dict={X:triples})
    
    '''
    To be replaced by inherited methods:
    '''
    
    def save(self, filename, variables):
        store_package = (variables[0],
                         variables[1],
                         variables[2],
                         variables[3],
                         self.n_entities,
                         self.n_relations)

        store_file = open(filename, 'wb')
        pickle.dump(store_package, store_file)
        store_file.close()

    def load(self, filename):
        store_file = open(filename, 'rb')
        store_package = pickle.load(store_file)

        self.W_embedding = tf.Variable(store_package[0])
        self.W_relation = tf.Variable(store_package[1])
        self.H1 = tf.Variable(store_package[2])
        self.H2 = tf.Variable(store_package[3])
        self.n_entities = store_package[4]
        self.n_relations = store_package[5]

        self.model_path = filename

