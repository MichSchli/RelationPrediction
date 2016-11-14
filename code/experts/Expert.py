import pickle
import theano
'''
Class representing an (encoder, decoder)-pair.
'''

class Expert():

    '''
    Fields
    '''
    model_path = None
    train_triplets = None
    valid_triplets = None
    test_triplets = None
    entity_count = None
    relation_count = None
    
    '''
    Initialization. Requires Encoder, Decoder, and Auxilliaries class.
    '''

    def __init__(self, encoder, decoder, optimizer_settings):
        self.encoder = encoder
        self.decoder = decoder

        self.settings = optimizer_settings

        
    '''
    Methods for setting up train and test processes:
    '''

    def set_model_path(self, model_path):
        self.model_path = model_path

    def preprocess(self, train_triplets, valid_triplets):
        self.train_triplets = train_triplets
        self.valid_triplets = valid_triplets

        self.encoder.preprocess(train_triplets)
        self.decoder.preprocess(train_triplets)

    def initialize_train(self):
        self.encoder.initialize_train()
        self.decoder.initialize_train()

    def initialize_test(self):
        self.encoder.initialize_test()
        self.decoder.initialize_test()


    
    '''
    Methods to return handles for weights and input.
    '''

    def get_additional_ops(self):
        return self.encoder.get_additional_ops() + self.decoder.get_additional_ops()
    
    def get_train_input_variables(self):
        return self.encoder.get_input_variables() + [self.decoder.get_gold_input_variable()]
    
    def get_test_input_variables(self):
        return self.encoder.get_input_variables()

    def get_weights(self):
        return self.encoder.get_weights() + self.decoder.get_weights()

    def get_weight_values(self):
        weights = self.get_weights()

        if self.settings['Backend'] == 'tensorflow':
            return self.session.run(weights)
        elif self.settings['Backend'] == 'theano':
            return [w.get_value() for w in weights]

    def assign_weights(self, weights):
        split = self.encoder.parameter_count()
        self.encoder.assign_weights(weights[:split])
        self.decoder.assign_weights(weights[split:])
    
    '''
    Call on components to compute a train-time loss for optimization. 
    Includes e.g. dropouts and regularization.
    '''

    def get_train_loss(self):
        code = self.encoder.encode(training=True)
        loss = self.decoder.loss(code)
        
        regularization = self.encoder.get_regularization()
        regularization += self.decoder.get_regularization()

        return loss + regularization

    
    '''
    Methods for quickly scoring all triples with respectively subject and object replaced:
    '''

    def get_all_subject_scores(self):
        code = self.encoder.encode(training=False)
        all_subject_codes = self.encoder.get_all_subject_codes()
        return self.decoder.fast_decode_all_subjects(code, all_subject_codes)

    def get_all_object_scores(self):
        code = self.encoder.encode(training=False)
        all_object_codes = self.encoder.get_all_object_codes()
        return self.decoder.fast_decode_all_objects(code, all_object_codes)

    score_all_subjects_graph = None
    score_all_objects_graph = None

    #Hacky
    def score_all_subjects(self, triples):
        if self.settings['Backend'] == 'tensorflow':
            return self.session.run(self.get_all_subject_scores(), feed_dict={self.encoder.X:triples})
        elif self.settings['Backend'] == 'theano':
            if self.score_all_subjects_graph is None:
                self.score_all_subjects_graph = theano.function(inputs=self.get_test_input_variables(), outputs=self.get_all_subject_scores())

            return self.score_all_subjects_graph(triples)

    def score_all_objects(self, triples):
        if self.settings['Backend'] == 'tensorflow':
            return self.session.run(self.get_all_object_scores(), feed_dict={self.encoder.X:triples})
        elif self.settings['Backend'] == 'theano':
            if self.score_all_objects_graph is None:
                self.score_all_objects_graph = theano.function(inputs=self.get_test_input_variables(), outputs=self.get_all_object_scores())

            return self.score_all_objects_graph(triples)
    
    '''
    Model persistence methods:
    '''

    def save(self, filename):
        store_package = self.get_weight_values()
        
        store_package += [self.entity_count,
                          self.relation_count]

        store_file = open(filename, 'wb')
        pickle.dump(store_package, store_file)
        store_file.close()

    def load(self, filename):
        store_file = open(filename, 'rb')
        store_package = pickle.load(store_file)

        self.assign_weights(store_package[:-2])
        self.n_entities = store_package[-2]
        self.n_relations = store_package[-1]
