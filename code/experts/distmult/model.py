from theano import tensor as T
import theano
import numpy as np
import pickle
import random
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

    
class Model():

    n_entities = None
    n_relations = None
    predict_function = None
    embedding_width = 100

    positives_forward = None
    positives_backward = None
    srng = None
    
    def __init__(self):
        pass

    def predict(self, triplets):        
        e1s, e2s, rels = self.expand_triplets(triplets)
        return self.wrapper_predict(e1s, e2s, rels)

    
    def process_train_triplets(self, triplet_sample, all_triplets, disable_saving=False):
        e1s, e2s, rels = self.expand_triplets(triplet_sample)
        c_e1s, c_e2s = self.generate_corrupted_entries(triplet_sample, all_triplets, self.positives_forward is None or disable_saving)

        if disable_saving:
            self.positives_forward = None
            self.positives_backward = None

        return e1s, e2s, rels, c_e1s, c_e2s

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
    
    
    def generate_corrupted_entries(self, triplets_to_corrupt, triplets_in_kb, generate_new_dicts=False):
        if generate_new_dicts:
            self.positives_forward, self.positives_backward = self.generate_positive_sample_dictionaries(triplets_in_kb)

        left_entries = np.zeros(len(triplets_to_corrupt)).astype(np.int32)
        right_entries = np.zeros(len(triplets_to_corrupt)).astype(np.int32)
        total = range(self.n_entities)
        
        for i,triplet in enumerate(triplets_to_corrupt):
            positive_subjects = self.positives_backward[triplet[2]][triplet[1]]
            positive_objects = self.positives_forward[triplet[0]][triplet[1]]

            # This may be ineffective if there exists an element with a relation to MANY other elements.
            # For now, just run las vegas:

            right_entry = None
            while right_entry is None:
                sample = random.choice(total)
                if sample not in positive_objects:
                    right_entry = sample

            left_entry = None
            while left_entry is None:
                sample = random.choice(total)
                if sample not in positive_subjects:
                    left_entry = sample
                    
            #negative_subject_space = np.setdiff1d(total, positive_subjects)
            #negative_object_space = np.setdiff1d(total, positive_objects)
            
            #print(len(negative_subject_space))
            left_entries[i] = left_entry
            right_entries[i] = right_entry
        
        return left_entries, right_entries

    def expand_triplets(self, triplets):
        triplet_array = np.array(triplets).astype(np.int32)
        organized = np.transpose(triplet_array)
        return organized[0], organized[2], organized[1]
    
    def set_entity_count(self, count):
        self.n_entities = count

    def set_relation_count(self, count):
        self.n_relations = count

    def glorot_initialization(self, n_from, n_to):
        value = np.sqrt(6)/ np.sqrt(n_from + n_to)
        return np.random.uniform(low=-value, high=value, size=(n_to, n_from)).astype(np.float32)
        
    def initialize_variables(self):
        embedding_initial = self.glorot_initialization(self.n_entities+1, self.embedding_width)

        relation_initial = np.random.normal(0, 1, size=(self.n_relations, self.embedding_width)).astype(np.float32)
        
        self.W_embedding = theano.shared(embedding_initial)
        self.W_relation = theano.shared(relation_initial)

    def get_weight_shapes(self):
        return ((self.embedding_width, self.n_entities+1), (self.n_relations, self.embedding_width))
        
    def get_weights(self):
        return (self.W_embedding,self.W_relation)
    
    def print_status(self):
        print(self.n_entities)
        print(self.n_relations)

    def nonlinearity(self, v):
        return 1.7159*T.tanh(0.66667*v)

    def get_update_list(self, update):
        embedding_weight_length = T.sqrt(T.sum(update[0]*update[0], axis=1))
        update[0] = update[0] / embedding_weight_length
        return [(self.W_embedding, update[0]),
                (self.W_relation, update[1])]

    def theano_l2_regularization(self):
        sum_of_squares = T.sum(self.W_embedding * self.W_embedding)
        sum_of_squares += T.sum(self.W_relation * self.W_relation)
        return T.sqrt(sum_of_squares)

    def theano_batch_predict(self, e1s, e2s, rs):
        e1s = T.extra_ops.to_one_hot(e1s, self.n_entities)
        e2s = T.extra_ops.to_one_hot(e2s, self.n_entities)

        e1s_embed = self.apply_embedding(e1s)
        e2s_embed = self.apply_embedding(e2s)
        
        conv_relations = self.expand_relations(rs)
        
        PT = e1s_embed * conv_relations * e2s_embed
        return T.sum(PT, axis=1)


    def apply_embedding(self, v):
        with_bias = T.concatenate([T.ones((v.shape[0], 1), dtype=v.dtype), v], axis=1)
        return self.nonlinearity(T.dot(with_bias, self.W_embedding.transpose())) 

    def expand_relations(self, relation_id_list, with_dropout=False):
        '''
        if with_dropout:
            if self.srng is None:
                self.srng = RandomStreams(seed=12345)

            mask = self.srng.binomial(n=1, p=0.5, size=self.W_relation.shape, dtype='float32')
            W = self.W_relation * mask
        else:
            W = self.W_relation * 0.5
        '''

        W = self.W_relation
        one_hot_representation = T.extra_ops.to_one_hot(relation_id_list, self.n_relations)
        return T.dot(one_hot_representation, W)

    def theano_batch_loss(self, input_variable_list):
        e1s = T.extra_ops.to_one_hot(input_variable_list[0], self.n_entities)
        e2s = T.extra_ops.to_one_hot(input_variable_list[1], self.n_entities)
        relations = input_variable_list[2]
        c_e1s = T.extra_ops.to_one_hot(input_variable_list[3], self.n_entities)
        c_e2s = T.extra_ops.to_one_hot(input_variable_list[4], self.n_entities)

        e1s_embed = self.apply_embedding(e1s)
        e2s_embed = self.apply_embedding(e2s)
        c_e1s_embed = self.apply_embedding(c_e1s)
        c_e2s_embed = self.apply_embedding(c_e2s)

        conv_relations = self.expand_relations(relations, with_dropout=True)
        
        PT = e1s_embed * conv_relations
        PT_corrupt = c_e1s_embed * conv_relations
        
        Uncorrupt = T.sum(PT * e2s_embed, axis=1)
        Corrupt_lhs = T.sum(PT_corrupt * e2s_embed, axis=1)
        Corrupt_rhs = T.sum(PT * c_e2s_embed, axis=1)
        
        l_c1 = T.maximum(Corrupt_lhs - Uncorrupt + 1, 0)
        l_c2 = T.maximum(Corrupt_rhs - Uncorrupt + 1, 0)

        return T.sum(l_c1 + l_c2)
    

    '''
    To be replaced by inherited methods:
    '''

    def wrapper_predict(self, e1_onehot, e2_onehot, relation_id):
        if self.predict_function is None:
            E1s = T.ivector('E1s')
            E2s = T.ivector('E2s')
            Rs = T.ivector('Rs')
        
            input_variable_list = [E1s, E2s, Rs]
            result = self.theano_batch_predict(E1s, E2s, Rs)

            self.predict_function = theano.function(inputs=input_variable_list, outputs=result)

        return self.predict_function(e1_onehot, e2_onehot, relation_id)

    def compute_batch_loss_function(self):
        input_variable_list = self.get_theano_input_variables()
        loss = self.theano_batch_loss(input_variable_list)
        return theano.function(inputs=input_variable_list, outputs=loss)

    
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
        E1s = T.vector('E1s', dtype='int32')
        E2s = T.vector('E2s', dtype='int32')

        CorruptedE1s = T.vector('E1s', dtype='int32')
        CorruptedE2s = T.vector('E1s', dtype='int32')

        Rs = T.vector('Rs', dtype='int32')

        return [E1s, E2s, Rs, CorruptedE1s, CorruptedE2s]
    
