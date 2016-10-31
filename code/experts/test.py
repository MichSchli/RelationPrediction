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
algorithm = imp.load_source('algorithm', 'code/experts/'+args.algorithm+'/model.py')
evaluation = imp.load_source('evaluation', 'code/evaluation/evaluation.py')

train_triplets = io.read_triplets_as_list(args.train_data, args.entities, args.relations)
valid_triplets = io.read_triplets_as_list(args.validation_data, args.entities, args.relations)
test_triplets = io.read_triplets_as_list(args.test_data, args.entities, args.relations)

scorer = evaluation.Scorer()

scorer.register_data(train_triplets)
scorer.register_data(valid_triplets)
scorer.register_data(test_triplets)

model = algorithm.Model()
model.session = tf.Session()
model.load(args.model_path)
model.initiate_eval()
scorer.register_model(model)

triples = np.array(test_triplets, dtype=np.int32)

scores = scorer.compute_scores(triples, verbose=True)
scores.summarize()