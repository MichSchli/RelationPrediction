import numpy as np
import pickle
import random
import imp
import tensorflow as tf

abstract_model = imp.load_source('abstract_model', 'code/experts/AbstractModel.py')
shared = imp.load_source('shared', 'code/experts/shared.py')
evaluation = imp.load_source('evaluation', 'code/evaluation/evaluation.py')

'''
Variational Distmult
'''
class Model():

    '''
    Fields:
    '''
    
    model_path = None
    backend = "tensorflow"
    
    n_entities = None
    n_relations = None

    batch_size = 5000
    embedding_width = 200
    number_of_negative_samples = 10
    regularization_parameter = 0.01
    variational_samples = 1

    e_mu_prior = np.zeros(embedding_width, dtype=np.float32)
    e_sigma_prior = np.ones(embedding_width, dtype=np.float32)

    r_mu_prior = np.zeros(embedding_width, dtype=np.float32)
    r_sigma_prior = np.ones(embedding_width, dtype=np.float32)

    positives_forward = None
    positives_backward = None

    session = None
    
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

    def score_validation_data(self, validation_data):
        scorer = evaluation.Scorer()
        scorer.register_data(self.graph_edges)
        scorer.register_data(validation_data)
        scorer.register_model(self)

        score_summary = scorer.compute_scores(validation_data, verbose=True).get_summary()
        return score_summary.results['Filtered'][score_summary.mrr_string()]
    
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
                ('TrainLossReporter', {'evaluate_every_n':10}),
                ('EarlyStopper', {'criteria':'score_validation_data',
                                  'evaluate_every_n':100,
                                  'scoring_function':self.score_validation_data,
                                  'comparator':lambda current, prev: current > prev,
                                  'burnin':20000}),
                ('ModelSaver', {'save_function': self.save,
                                'model_path': self.model_path,
                                'save_every_n':100})]
    
    def initialize_variables(self):
        self.embedding_initial_mu = np.random.randn(self.n_entities, self.embedding_width).astype(np.float32)
        self.embedding_initial_sigma = np.random.randn(self.n_entities, self.embedding_width).astype(np.float32)
        self.relation_initial_mu = np.random.randn(self.n_relations, self.embedding_width).astype(np.float32)
        self.relation_initial_sigma = np.random.randn(self.n_relations, self.embedding_width).astype(np.float32)

        self.embedding_mu_initial_bias = np.random.randn(self.embedding_width).astype(np.float32)
        self.embedding_sigma_initial_bias = np.random.randn(self.embedding_width).astype(np.float32)
        self.relation_mu_initial_bias = np.random.randn(self.embedding_width).astype(np.float32)
        self.relation_sigma_initial_bias = np.random.randn(self.embedding_width).astype(np.float32)

        self.X = tf.placeholder(tf.int32, shape=[None,3])
        self.Y = tf.placeholder(tf.float32, shape=[None])

        self.W_embedding_mu = tf.Variable(self.embedding_initial_mu)
        self.W_embedding_sigma = tf.Variable(self.embedding_initial_sigma)
        self.W_relation_mu = tf.Variable(self.relation_initial_mu)
        self.W_relation_sigma = tf.Variable(self.relation_initial_sigma)

        self.b_embedding_mu = tf.Variable(self.embedding_mu_initial_bias)
        self.b_embedding_sigma = tf.Variable(self.embedding_sigma_initial_bias)
        self.b_relation_mu = tf.Variable(self.relation_mu_initial_bias)
        self.b_relation_sigma = tf.Variable(self.relation_sigma_initial_bias)

        
    def tf_diag_mvn_kl_div(self, mu1, sigma1, mu2, sigma2):
        logsig = tf.log(sigma2 / sigma1)
        v = (tf.square(sigma1)+tf.square(mu1 - mu2))/(2*tf.square(sigma2)) - 0.5
        return tf.reduce_mean(tf.reduce_sum(logsig + v, 1), 0)
        

    def get_optimizer_loss(self):
        full_batch_size = tf.shape(self.Y)[0]

        e1_sample = tf.random_normal((self.variational_samples, full_batch_size, self.embedding_width), 0, 1)
        r_sample = tf.random_normal((self.variational_samples, full_batch_size, self.embedding_width), 0, 1)
        e2_sample = tf.random_normal((self.variational_samples, full_batch_size, self.embedding_width), 0, 1)
        
        e1s_mu = tf.nn.embedding_lookup(self.W_embedding_mu, self.X[:,0])+self.b_embedding_mu
        e1s_sigma = tf.exp(tf.nn.embedding_lookup(self.W_embedding_sigma, self.X[:,0])+self.b_embedding_sigma)
        
        rs_mu = tf.nn.embedding_lookup(self.W_relation_mu, self.X[:,1])+self.b_relation_mu
        rs_sigma = tf.exp(tf.nn.embedding_lookup(self.W_relation_sigma, self.X[:,1])+self.b_relation_sigma)

        e2s_mu = tf.nn.embedding_lookup(self.W_embedding_mu, self.X[:,2])+self.b_embedding_mu
        e2s_sigma = tf.exp(tf.nn.embedding_lookup(self.W_embedding_sigma, self.X[:,2])+self.b_embedding_sigma)

        e1s_sampled = e1s_mu + 0*tf.mul(e1s_sigma, e1_sample)
        rs_sampled = rs_mu + 0*tf.mul(rs_sigma, r_sample)
        e2s_sampled = e2s_mu + 0*tf.mul(e2s_sigma, e2_sample)

        energies = tf.reduce_sum(e1s_sampled*rs_sampled*e2s_sampled, 2)

        exp_ys = tf.expand_dims(self.Y, 0)
        expanded_ys = tf.tile(exp_ys, [self.variational_samples, 1])

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(energies, expanded_ys))

        kl = self.tf_diag_mvn_kl_div(e1s_mu, e1s_sigma, self.e_mu_prior, self.e_sigma_prior)
        kl += self.tf_diag_mvn_kl_div(rs_mu, rs_sigma, self.r_mu_prior, self.r_sigma_prior)
        kl += self.tf_diag_mvn_kl_div(e2s_mu, e2s_sigma, self.e_mu_prior, self.e_sigma_prior)
        
        regularization = tf.reduce_mean(tf.square(e1s_sampled)) + tf.reduce_mean(tf.square(rs_sampled)) + tf.reduce_mean(tf.square(e2s_sampled))
        return loss + self.regularization_parameter * regularization

    def get_prediction(self, X):
        e1s = tf.nn.embedding_lookup(self.W_embedding_mu, X[:,0])
        rs = tf.nn.embedding_lookup(self.W_relation_mu, X[:,1])
        e2s = tf.nn.embedding_lookup(self.W_embedding_mu, X[:,2])

        energies = tf.reduce_sum(e1s*rs*e2s, 1)
        return tf.nn.sigmoid(energies)
    
    def get_optimizer_weights(self):
        weights = [self.W_embedding_mu, self.W_embedding_sigma, self.W_relation_mu, self.W_relation_sigma]
        biases = [self.b_embedding_mu, self.b_embedding_sigma, self.b_relation_mu, self.b_relation_sigma]

        return weights + biases
    
    #Fast, ugly, eval:

    def initiate_eval(self):
        self.X = tf.placeholder(tf.int32, shape=[None,3])
        
        init_op = tf.initialize_all_variables()

        self.session.run(init_op)

    def compute_o(self):
        e1s = tf.nn.embedding_lookup(self.W_embedding_mu, self.X[:,0]) + self.b_embedding_mu
        rs = tf.nn.embedding_lookup(self.W_relation_mu, self.X[:,1]) + self.b_relation_mu
        
        thingy2 = tf.matmul(e1s*rs, tf.transpose(self.W_embedding_mu + self.b_embedding_mu))
        return tf.nn.sigmoid(thingy2)
        
    def compute_s(self):
        rs = tf.nn.embedding_lookup(self.W_relation_mu, self.X[:,1]) + self.b_embedding_mu
        e2s = tf.nn.embedding_lookup(self.W_embedding_mu, self.X[:,2]) + self.b_relation_mu

        thingy1 = tf.transpose(tf.matmul(self.W_embedding_mu + self.b_embedding_mu, tf.transpose(rs*e2s)))
        return tf.nn.sigmoid(thingy1)
    
    def score_all_subjects(self, tup):
        #sess = tf.Session()
        return self.session.run(self.compute_s(), feed_dict={self.X:tup})

    def score_all_objects(self, tup):
        #sess = tf.Session()
        #init_op = tf.initialize_all_variables()

        #sess.run(init_op)
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
        store_package = self.session.run([self.W_embedding_mu,
                                          self.W_embedding_sigma,
                                          self.W_relation_mu,
                                          self.W_relation_sigma])

        store_package += [self.n_entities,
                         self.n_relations]

        store_file = open(filename, 'wb')
        pickle.dump(store_package, store_file)
        store_file.close()

    def load(self, filename):
        store_file = open(filename, 'rb')
        store_package = pickle.load(store_file)

        self.W_embedding_mu = tf.Variable(store_package[0])
        self.W_embedding_sigma = tf.Variable(store_package[1])
        self.W_relation_mu = tf.Variable(store_package[2])
        self.W_relation_sigma = tf.Variable(store_package[2])
        self.n_entities = store_package[3]
        self.n_relations = store_package[4]

        self.model_path = filename

        
    
