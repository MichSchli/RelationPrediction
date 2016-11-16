import imp
import argparse
import numpy as np
import Converge.optimize as Converge
import tensorflow as tf

parser = argparse.ArgumentParser(description="Train a model on a given dataset.")
parser.add_argument("--relations", help="Filepath for generated relation dictionary.", required=True)
parser.add_argument("--entities", help="Filepath for generated entity dictionary.", required=True)
parser.add_argument("--train_data", help="Filepath for formatted training data.", required=True)
parser.add_argument("--validation_data", help="Filepath for formatted validation data.", required=True)
parser.add_argument("--test_data", help="Filepath for formatted test data (only for filtering).", required=True)
parser.add_argument("--model_path", help="Filepath to store the trained model.", required=True)
parser.add_argument("--algorithm", help="Algorithm to train.", required=True)
args = parser.parse_args()

io = imp.load_source('io', 'code/common/io.py')
auxilliaries = imp.load_source('auxilliaries', 'code/common/auxilliaries.py')
evaluation = imp.load_source('evaluation', 'code/evaluation/evaluation.py')

'''
Load in the data:
'''

train_triplets = io.read_triplets_as_list(args.train_data, args.entities, args.relations)
valid_triplets = io.read_triplets_as_list(args.validation_data, args.entities, args.relations)
test_triplets = io.read_triplets_as_list(args.test_data, args.entities, args.relations)

entities = io.read_dictionary(args.entities)
relations = io.read_dictionary(args.relations)

'''
Handle settings:
'''

settings_reader = imp.load_source('settings_reader', 'code/common/settings_reader.py')
settings = settings_reader.read('settings/'+args.algorithm+'.exp')

encoder_settings = settings['Encoder']
decoder_settings = settings['Decoder']
shared_settings = settings['Shared']
general_settings = settings['General']
optimizer_settings = settings['Optimizer']

general_settings.put('EntityCount', len(entities))
general_settings.put('RelationCount', len(relations))
general_settings.put('EdgeCount', len(train_triplets))

encoder_settings.merge(shared_settings)
encoder_settings.merge(general_settings)
decoder_settings.merge(shared_settings)
decoder_settings.merge(general_settings)

optimizer_settings.put('ModelPath', args.model_path)
optimizer_settings.merge(general_settings)

'''
Construct the expert:
'''

encoder = imp.load_source('Encoder', 'code/experts/encoders/'+encoder_settings['Name']+'/encoder-'+general_settings['Backend']+'.py')
decoder = imp.load_source('Decoder', 'code/experts/decoders/'+decoder_settings['Name']+'/decoder-'+general_settings['Backend']+'.py')
expert = imp.load_source('Expert', 'code/experts/Expert.py')

encoder = encoder.Encoder(encoder_settings)
decoder = decoder.Decoder(decoder_settings)
expert = expert.Expert(encoder, decoder, optimizer_settings)

'''
Initialize for training:
'''

expert.preprocess(train_triplets, valid_triplets)
expert.initialize_train()

optimizer_weights = expert.get_weights()
optimizer_input = expert.get_train_input_variables()
loss = expert.get_train_loss()

'''
Get parameters for optimizer:
'''

optimizer_parameter_parsing = imp.load_source('optimizer_parameter_parser', 'code/common/optimizer_parameter_parser.py')
opp = optimizer_parameter_parsing.Parser(optimizer_settings)
opp.set_save_function(expert.save)

def score_validation_data(validation_data):
    scorer = evaluation.Scorer()
    scorer.register_data(train_triplets)
    scorer.register_data(valid_triplets)
    scorer.register_data(test_triplets)
    scorer.register_model(expert)

    score_summary = scorer.compute_scores(validation_data, verbose=True).get_summary()
    return score_summary.results['Filtered'][score_summary.mrr_string()]

opp.set_early_stopping_score_function(score_validation_data)

if 'NegativeSampleRate' in general_settings:
    ns = auxilliaries.NegativeSampler(int(general_settings['NegativeSampleRate']), general_settings['EntityCount'])
    opp.set_sample_transform_function(ns.transform)

opp.set_additional_ops(expert.get_additional_ops())
    
optimizer_parameters = opp.get_parametrization()

print(optimizer_parameters)
'''
Train with Converge:
'''

if general_settings['Backend'] == 'theano':
    optimizer = Converge.build_theano(loss, optimizer_weights, optimizer_parameters, optimizer_input)
elif general_settings['Backend'] == 'tensorflow':
    expert.session = tf.Session()
    optimizer = Converge.build_tensorflow(loss, optimizer_weights, optimizer_parameters, optimizer_input)
    optimizer.set_session(expert.session)
    
optimizer.fit(train_triplets, validation_data=valid_triplets)
