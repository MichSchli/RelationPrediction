from theano import tensor as T
import theano
import numpy as np
import pickle
import random


    
class Model():

    n_entities = None
    n_relations = None
    predict_function = None
    
    def __init__(self):
        pass

    def predict(self, triplets):
        e1s, e2s, rels = self.expand_triplets(triplets)
        return self.wrapper_predict(e1s, e2s, rels)

    
    def process_train_triplets(self, triplet_sample, all_triplets):
        e1s, e2s, rels = self.expand_triplets(triplet_sample)
        c_e1s, c_e2s = self.generate_corrupted_entries(triplet_sample, all_triplets)

        return e1s, e2s, rels, self.expand_entities(c_e1s), self.expand_entities(c_e2s)
    
    def generate_corrupted_entries(self, triplets_to_corrupt, triplets_in_kb):
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

        left_entries = [None]*len(triplets_to_corrupt)
        right_entries = [None]*len(triplets_to_corrupt)
        for i,triplet in enumerate(triplets_to_corrupt):
            negative_head_space = [e for e in range(self.n_entities) if not e in positives_forward[triplet[0]][triplet[1]]]
            negative_tail_space = [e for e in range(self.n_entities) if not e in positives_forward[triplet[0]][triplet[1]]]
            
            left_entries[i] = random.choice(negative_head_space)
            right_entries[i] = random.choice(negative_tail_space)
        
        return left_entries, right_entries

    def expand_triplets(self, triplets):
        e1s = np.zeros((len(triplets), self.n_entities)).astype(np.float32)
        e2s = np.zeros((len(triplets), self.n_entities)).astype(np.float32)
        rels = np.zeros(len(triplets)).astype(np.int32)

        for i,triplet in enumerate(triplets):
            e1s[i][triplet[0]] = 1
            e2s[i][triplet[2]] = 1
            rels[i] = triplet[1]

        return e1s, e2s, rels

    def expand_entities(self, entities):
        expanded = np.zeros((len(entities), self.n_entities)).astype(np.float32)
        
        for i, entity in enumerate(entities):
            expanded[i][entity] = 1
            
        return expanded
    
    def set_entity_count(self, count):
        self.n_entities = count

    def set_relation_count(self, count):
        self.n_relations = count

    def glorot_initialization(self, n_from, n_to):
        value = np.sqrt(6) / np.sqrt(n_from + n_to)
        return np.random.uniform(low=-value, high=value, size=(n_to, n_from)).astype(np.float32)
        
    def initialize_variables(self):
        embedding_initial = self.glorot_initialization(self.n_entities+1, 100)

        relation_initial = np.random.normal(0, 1, size=(self.n_relations, 100)).astype(np.float32)
        
        self.W_embedding = theano.shared(embedding_initial)
        self.W_relation = theano.shared(relation_initial)

    def get_weight_shapes(self):
        return ((100, self.n_entities+1), (self.n_relations, 100))
        
    def get_weights(self):
        return (self.W_embedding,self.W_relation)

    
    def print_status(self):
        print(self.n_entities)
        print(self.n_relations)
    
    def perceptron(self, W, v):
        with_bias = T.concatenate((v, T.ones(1, dtype='float32')))
        return T.dot(W, with_bias)

    def nonlinearity(self, v):
        return T.tanh(v)

    def get_update_list(self, update):
        return [(self.W_embedding, update[0]),
                (self.W_relation, update[1])]

    def theano_l2_regularization(self):
        sum_of_squares = T.sum(self.W_embedding * self.W_embedding)
        sum_of_squares += T.sum(self.W_relation * self.W_relation)
        return T.sqrt(sum_of_squares)
    
    def theano_loss(self, e1, e2, relation, c_e1, c_e2):
        p_gold = self.theano_predict(e1, e2, relation)
        p_c1 = self.theano_predict(c_e1, e2, relation)
        p_c2 = self.theano_predict(e1, c_e2, relation)

        l_c1 = T.max((p_c1 - p_gold + 1, 0))
        l_c2 = T.max((p_c2 - p_gold + 1, 0))

        return l_c1 + l_c2

    def theano_predict(self, e1, e2, relation):
        embed_1 = self.nonlinearity(self.perceptron(self.W_embedding, e1))
        embed_2 = self.nonlinearity(self.perceptron(self.W_embedding, e2))

        W_diag = self.W_relation[relation]
        W = T.nlinalg.diag(W_diag)

        score = T.dot(T.transpose(embed_1), T.dot(W, embed_2))
        return score

    

    '''
    To be replaced by inherited methods:
    '''

    def wrapper_predict(self, e1_onehot, e2_onehot, relation_id):
        if self.predict_function is None:
            E1s = T.imatrix('E1s')
            E2s = T.imatrix('E2s')
            Rs = T.ivector('Rs')
        
            input_variable_list = [E1s, E2s, Rs]
            result,_ = theano.scan(self.theano_predict,
                                   sequences=input_variable_list)

            self.predict_function = theano.function(inputs=input_variable_list, outputs=result)


        return self.predict_function(e1_onehot, e2_onehot, relation_id)

    
    def compute_batch_loss_function(self):
        input_variable_list = self.get_theano_input_variables()
        loss,_ = theano.scan(self.theano_loss,
                             sequences=input_variable_list)

        sumloss = T.sum(loss)
        return theano.function(inputs=input_variable_list, outputs=sumloss)

    
    def save(self, filename):
        store_package = (self.W_embedding.get_value(),
                         self.W_relation.get_value(),
                         self.n_entities,
                         self.n_relations)

        store_file = open(filename, 'wb')
        pickle.dump(store_package, store_file)
        store_file.close()

    def load(self, filename):
        store_file = open(filename, 'rb')
        store_package = pickle.load(store_file)

        self.W_embedding = theano.shared(store_package[0])
        self.W_relation = theano.shared(store_package[1])
        self.n_entities = store_package[2]
        self.n_relations = store_package[3]
        
    
    def get_theano_input_variables(self):
        E1s = T.matrix('E1s', dtype='float32')
        E2s = T.matrix('E2s', dtype='float32')

        CorruptedE1s = T.matrix('E1s', dtype='float32')
        CorruptedE2s = T.matrix('E1s', dtype='float32')

        Rs = T.vector('Rs', dtype='int32')

        return [E1s, E2s, Rs, CorruptedE1s, CorruptedE2s]
    
