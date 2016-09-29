import imp
import argparse
import numpy as np


parser = argparse.ArgumentParser(description="Make predictions to evaluate a stored model.")
parser.add_argument("--relations", help="Filepath for generated relation dictionary.", required=True)
parser.add_argument("--entities", help="Filepath for generated entity dictionary.", required=True)
parser.add_argument("--train_data", help="Filepath for formatted training data.", required=True)
parser.add_argument("--validation_data", help="Filepath for formatted validation data.", required=True)
parser.add_argument("--test_data", help="Filepath for formatted training data.", required=True)
parser.add_argument("--model_path", help="Filepath to store the trained model.", required=True)
parser.add_argument("--algorithm", help="Algorithm to train.", required=True)
parser.add_argument("--prediction_file", help="Output filepath.", required=True)
args = parser.parse_args()


outfilepath = args.prediction_file

io = imp.load_source('io', 'code/common/io.py')
algorithm = imp.load_source('algorithm', 'code/experts/'+args.algorithm+'/model.py')

train_triplets = io.read_triplets_as_list(args.train_data, args.entities, args.relations)
valid_triplets = io.read_triplets_as_list(args.validation_data, args.entities, args.relations)
test_triplets = io.read_triplets_as_list(args.test_data, args.entities, args.relations)

entities = io.read_dictionary(args.entities)
relations = io.read_dictionary(args.relations)

extended_triplets = []

def remove(triplet):
    global train_triplets
    global valid_triplets
    global test_triplets

    return triplet in train_triplets or triplet in valid_triplets or triplet in test_triplets

def search_triplets(triplets, head):
    for i,triplet in enumerate(triplets):
        if head == triplet[0]:
            return i, triplet[1]
    return -1,-1

positives = {}
for triplet in train_triplets:
    if triplet[0] not in positives:
        positives[triplet[0]] = {triplet[1] : [triplet[2]]}
    else:
        if triplet[1] not in positives[triplet[0]]:
            positives[triplet[0]][triplet[1]] = [triplet[2]]
        else:
            positives[triplet[0]][triplet[1]].append(triplet[2])

for triplet in valid_triplets:
    if triplet[0] not in positives:
        positives[triplet[0]] = {triplet[1] : [triplet[2]]}
    else:
        if triplet[1] not in positives[triplet[0]]:
            positives[triplet[0]][triplet[1]] = [triplet[2]]
        else:
            positives[triplet[0]][triplet[1]].append(triplet[2])

for triplet in test_triplets:
    if triplet[0] not in positives:
        positives[triplet[0]] = {triplet[1] : [triplet[2]]}
    else:
        if triplet[1] not in positives[triplet[0]]:
            positives[triplet[0]][triplet[1]] = [triplet[2]]
        else:
            positives[triplet[0]][triplet[1]].append(triplet[2])

            
def included(dic, triplet):
    if triplet[0] not in dic:
        return False
    if triplet[1] not in dic[triplet[0]]:
        return False
    else:
        return triplet[2] in dic[triplet[0]][triplet[1]]


model = algorithm.Model()
model.load(args.model_path)

outfile = open(outfilepath, 'w+')
for i,triplet in enumerate(test_triplets):
    print(i)
    extended_triplets = [[i, triplet[1], triplet[2]] for i in range(len(entities))]
    extended_triplets += [[triplet[0], triplet[1], i] for i in range(len(entities))]    

    positives_object_replaced = positives[triplet[0]][triplet[1]]
    positives_subject_replaced = [i for i in range(len(entities)) if i in positives and triplet[1] in positives[i] and triplet[2] in positives[i][triplet[1]]]

    #print(len(positives_object_replaced))
    #print(len(positives_subject_replaced))
    predictions = model.predict(extended_triplets)
    
    score_gold = predictions[triplet[0]]
    print(score_gold)
    
    gold_raw_rank = 1
    gold_filtered_rank = 1

    subjects_better = np.sum(predictions[:len(entities)] >= score_gold)
    objects_better = np.sum(predictions[len(entities):] >= score_gold)

    subjects_better_filtered = subjects_better - np.sum(predictions[:len(entities)][positives_subject_replaced] >= score_gold)
    objects_better_filtered = objects_better - np.sum(predictions[len(entities):][positives_object_replaced] >= score_gold)

    print(subjects_better, objects_better, subjects_better_filtered, objects_better_filtered)
    
    print(str(subjects_better + 1) +
          '\t'+ str(subjects_better_filtered + 1) +
          '\t'+ str(score_gold)
          , file=outfile)

    print(str(objects_better + 1) +
          '\t'+ str(objects_better_filtered + 1) +
          '\t'+ str(score_gold)
          , file=outfile)
