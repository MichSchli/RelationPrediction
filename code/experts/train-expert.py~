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
algorithm = imp.load_source('algorithm', 'code/experts/'+args.algorithm+'/model.py')

train_triplets = io.read_triplets_as_list(args.train_data, args.entities, args.relations)
valid_triplets = io.read_triplets_as_list(args.validation_data, args.entities, args.relations)
test_triplets = io.read_triplets_as_list(args.test_data, args.entities, args.relations)

entities = io.read_dictionary(args.entities)
relations = io.read_dictionary(args.relations)

print(args.model_path)
model = algorithm.Model()
model.set_entity_count(len(entities))
model.set_relation_count(len(relations))
model.initialize_variables()

model.set_model_path(args.model_path)
#model.load(args.model_path)

model.preprocess(train_triplets)

#CAREFUL
model.set_test_data(test_triplets)

optimizer_parameters = model.get_optimizer_parameters()
optimizer_weights = model.get_optimizer_weights()
optimizer_input = model.get_optimizer_input_variables()
loss = model.get_optimizer_loss()

if model.backend == 'theano':
    optimizer = Converge.build(loss, optimizer_weights, optimizer_parameters, optimizer_input)
elif model.backend == 'tensorflow':
    model.session = tf.Session()
    optimizer = Converge.tfbuild(loss, optimizer_weights, optimizer_parameters, optimizer_input)
    optimizer.set_session(model.session)
    
optimizer.fit(train_triplets, validation_data=valid_triplets)
