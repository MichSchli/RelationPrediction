import argparse
import random

import tensorflow as tf
from optimization.optimize import build_tensorflow
from common import settings_reader, io, model_builder, optimizer_parameter_parser, evaluation, auxilliaries
from model import Model
import numpy as np

# https://stackoverflow.com/questions/53429896/how-do-i-disable-tensorflows-eager-execution
tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser(description="Train a model on a given dataset.")
parser.add_argument("--settings", help="Filepath for settings file.", required=True)
parser.add_argument("--dataset", help="Filepath for dataset.", required=True)
args = parser.parse_args()

settings = settings_reader.read(args.settings)



'''
Load datasets:
'''

dataset = args.dataset

relations_path = dataset + '/relations.dict'
entities_path = dataset + '/entities.dict'
train_path = dataset + '/train.txt'
valid_path = dataset + '/valid.txt'
test_path = dataset + '/test.txt'

#Extend paths for accuracy evaluation:
if settings['Evaluation']['Metric'] == 'Accuracy':
    valid_path = dataset + '/valid_accuracy.txt'
    test_path = dataset + '/test_accuracy.txt'

train_triplets = io.read_triplets_as_list(train_path, entities_path, relations_path)

valid_triplets = io.read_triplets_as_list(valid_path, entities_path, relations_path)
test_triplets = io.read_triplets_as_list(test_path, entities_path, relations_path)


train_triplets = np.array(train_triplets)
valid_triplets = np.array(valid_triplets)
test_triplets = np.array(test_triplets)

entities = io.read_dictionary(entities_path)
relations = io.read_dictionary(relations_path)

'''
shuffled_rels = np.arange(len(relations))
np.random.shuffle(shuffled_rels)

known_rels = shuffled_rels[:int(len(relations)/2)]
target_rels = shuffled_rels[int(len(relations)/2):]

known_train = train_triplets[np.where(np.in1d(train_triplets[:,1], known_rels))]
target_train = train_triplets[np.where(np.in1d(train_triplets[:,1], target_rels))]
known_valid = valid_triplets[np.where(np.in1d(valid_triplets[:,1], known_rels))]
target_valid = valid_triplets[np.where(np.in1d(valid_triplets[:,1], target_rels))]
known_test = test_triplets[np.where(np.in1d(test_triplets[:,1], known_rels))]
target_test = test_triplets[np.where(np.in1d(test_triplets[:,1], target_rels))]
'''

'''
Load general settings
'''

encoder_settings = settings['Encoder']
decoder_settings = settings['Decoder']
shared_settings = settings['Shared']
general_settings = settings['General']
optimizer_settings = settings['Optimizer']
evaluation_settings = settings['Evaluation']

general_settings.put('EntityCount', len(entities))
general_settings.put('RelationCount', len(relations))
general_settings.put('EdgeCount', len(train_triplets))

encoder_settings.merge(shared_settings)
encoder_settings.merge(general_settings)
decoder_settings.merge(shared_settings)
decoder_settings.merge(general_settings)

optimizer_settings.merge(general_settings)
evaluation_settings.merge(general_settings)


'''
Construct the encoder-decoder pair:
'''
encoder = model_builder.build_encoder(encoder_settings, train_triplets, entities)
model = model_builder.build_decoder(encoder, decoder_settings)

'''
Construct the optimizer with validation MRR as early stopping metric:
'''

opp = optimizer_parameter_parser.Parser(optimizer_settings)
opp.set_save_function(model.save)

scorer = evaluation.Scorer(evaluation_settings)
scorer.register_data(train_triplets)
scorer.register_data(valid_triplets)
scorer.register_data(test_triplets)
scorer.register_degrees(train_triplets)
scorer.register_model(model)
scorer.finalize_frequency_computation(np.concatenate((train_triplets, valid_triplets, test_triplets), axis=0))

def score_validation_data(validation_data):
    score_summary = scorer.compute_scores(validation_data, verbose=False).get_summary()
    #score_summary.dump_degrees('dumps/degrees.in', 'dumps/degrees.out')
    #score_summary.dump_frequencies('dumps/near.freq', 'dumps/target.freq')
    #score_summary.pretty_print()

    if evaluation_settings['Metric'] == 'MRR':
        lookup_string = score_summary.mrr_string()
    elif evaluation_settings['Metric'] == 'Accuracy':
        lookup_string = score_summary.accuracy_string()

    early_stopping = score_summary.results['Filtered'][lookup_string]

    score_summary = scorer.compute_scores(test_triplets, verbose=False).get_summary()
    score_summary.pretty_print()

    return early_stopping


opp.set_early_stopping_score_function(score_validation_data)

print(len(train_triplets))

adj_list = [[] for _ in entities]
for i,triplet in enumerate(train_triplets):
    adj_list[triplet[0]].append([i, triplet[2]])
    adj_list[triplet[2]].append([i, triplet[0]])

degrees = np.array([len(a) for a in adj_list])
adj_list = [np.array(a) for a in adj_list]


def sample_TIES(triplets, n_target_vertices):
    vertex_set = set([])

    edge_indices = np.arange(triplets.shape[0])
    while len(vertex_set) < n_target_vertices:
        edge = triplets[np.random.choice(edge_indices)]
        new_vertices = [edge[0], edge[1]]
        vertex_set = vertex_set.union(new_vertices)

    sampled = [False]*triplets.shape[0]

    for i in edge_indices:
        edge = triplets[i]
        if edge[0] in vertex_set and edge[2] in vertex_set:
            sampled[i] = True

    return edge_indices[sampled]


def sample_edge_neighborhood(triplets, sample_size):

    edges = np.zeros((sample_size), dtype=np.int32)

    #initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in triplets])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]), p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges


if 'NegativeSampleRate' in general_settings:
    ns = auxilliaries.NegativeSampler(int(general_settings['NegativeSampleRate']), general_settings['EntityCount'])
    ns.set_known_positives(train_triplets)

    def t_func(x): #horrible hack!!!
        arr = np.array(x)
        if not encoder.needs_graph():
            return ns.transform(arr)
        else:
            if 'GraphBatchSize' in general_settings:
                graph_batch_size = int(general_settings['GraphBatchSize'])

                '''
                n = np.zeros(100)
                for i in range(100):
                    if i % 20 == 0:
                        print(i)
                    n[i] = sample_TIES(arr, 1000).shape[0]

                print(n.mean())
                print(n.std())
                exit()
                '''


                #graph_batch_ids = sample_TIES(arr, 1000) #sample_edge_neighborhood(arr, graph_batch_size)
                graph_batch_ids = sample_edge_neighborhood(arr, graph_batch_size)
            else:
                graph_batch_size = arr.shape[0]
                graph_batch_ids = np.arange(graph_batch_size)

            graph_batch = np.array(train_triplets)[graph_batch_ids]

            # Apply dropouts:
            graph_percentage = float(general_settings['GraphSplitSize'])
            split_size = int(graph_percentage * graph_batch.shape[0])
            graph_split_ids = np.random.choice(graph_batch_ids, size=split_size, replace=False)
            graph_split = np.array(train_triplets)[graph_split_ids]

            t = ns.transform(graph_batch)

            if 'StoreEdgeData' in encoder_settings and encoder_settings['StoreEdgeData'] == "Yes":
                return (graph_split, graph_split_ids, t[0], t[1])
            else:
                return (graph_split, t[0], t[1])

    opp.set_sample_transform_function(t_func)


'''
Initialize for training:
'''

# Hack for validation evaluation:
model.preprocess(train_triplets)
model.register_for_test(train_triplets)

model.initialize_train()

optimizer_weights = model.get_weights()
optimizer_input = model.get_train_input_variables()
loss = model.get_loss(mode='train') + model.get_regularization()
print(optimizer_input)

'''
Clean this up:
'''

for add_op in model.get_additional_ops():
    opp.additional_ops.append(add_op)

optimizer_parameters = opp.get_parametrization()

'''
Train with Converge:
'''

model.session = tf.compat.v1.Session()
optimizer = build_tensorflow(loss, optimizer_weights, optimizer_parameters, optimizer_input)
optimizer.set_session(model.session)

optimizer.fit(train_triplets, validation_data=valid_triplets)
#scorer.dump_all_scores(valid_triplets, 'dumps/subjects.valid', 'dumps/objects.valid')
#scorer.dump_all_scores(test_triplets, 'dumps/subjects.test', 'dumps/objects.test')
