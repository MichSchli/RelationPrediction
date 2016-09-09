import imp
import argparse
import numpy as np
from optimizers import RmsProp as Optimizer

parser = argparse.ArgumentParser(description="Train a model on a given dataset.")
parser.add_argument("--relations", help="Filepath for generated relation dictionary.", required=True)
parser.add_argument("--entities", help="Filepath for generated entity dictionary.", required=True)
parser.add_argument("--train_data", help="Filepath for formatted training data.", required=True)
parser.add_argument("--validation_data", help="Filepath for formatted validation data.", required=True)
parser.add_argument("--model_path", help="Filepath to store the trained model.", required=True)
parser.add_argument("--algorithm", help="Algorithm to train.", required=True)
args = parser.parse_args()

io = imp.load_source('io', 'code/common/io.py')
algorithm = imp.load_source('algorithm', 'code/experts/'+args.algorithm+'/model.py')

train_triplets = io.read_triplets_as_list(args.train_data, args.entities, args.relations)
valid_triplets = io.read_triplets_as_list(args.validation_data, args.entities, args.relations)

entities = io.read_dictionary(args.entities)
relations = io.read_dictionary(args.relations)

model = algorithm.Model()
model.set_entity_count(len(entities))
model.set_relation_count(len(relations))
model.initialize_variables()

optimizer = Optimizer()
optimizer.train(model, train_triplets, valid_triplets[:20000], args.model_path)

