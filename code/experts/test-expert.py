import imp
import argparse
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description="Make predictions to evaluate a stored model.")
parser.add_argument("--relations", help="Filepath for generated relation dictionary.", required=True)
parser.add_argument("--entities", help="Filepath for generated entity dictionary.", required=True)
parser.add_argument("--train_data", help="Filepath for formatted training data.", required=True)
parser.add_argument("--validation_data", help="Filepath for formatted validation data.", required=True)
parser.add_argument("--test_data", help="Filepath for formatted training data.", required=True)
parser.add_argument("--model_path", help="Filepath to store the trained model.", required=True)
parser.add_argument("--algorithm", help="Algorithm to train.", required=True)
args = parser.parse_args()

io = imp.load_source('io', 'code/common/io.py')
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

encoder_settings.merge(shared_settings)
encoder_settings.merge(general_settings)
decoder_settings.merge(shared_settings)
decoder_settings.merge(general_settings)


'''
Construct the expert:
'''

encoder = imp.load_source('Encoder', 'code/experts/encoders/'+encoder_settings['Name']+'/encoder.py')
decoder = imp.load_source('Decoder', 'code/experts/decoders/'+decoder_settings['Name']+'/decoder.py')
expert = imp.load_source('Expert', 'code/experts/Expert.py')

encoder = encoder.Encoder(encoder_settings)
decoder = decoder.Decoder(decoder_settings)
expert = expert.Expert(encoder, decoder, optimizer_settings)
expert.preprocess(train_triplets, valid_triplets)

expert.load(args.model_path)
expert.initialize_test()

#Hacky
expert.session = tf.Session()
expert.session.run(tf.initialize_all_variables())

'''
Construct the scorer:
'''

scorer = evaluation.Scorer()

scorer.register_data(train_triplets)
scorer.register_data(valid_triplets)
scorer.register_data(test_triplets)
scorer.register_model(expert)

'''
Run evaluation:
'''

triples = np.array(test_triplets, dtype=np.int32)
scores = scorer.compute_scores(triples, verbose=True)
scores.summarize()
