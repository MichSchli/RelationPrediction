import argparse
import tensorflow as tf
from Converge.optimize import build_tensorflow
from common import settings_reader, io, model_builder, optimizer_parameter_parser, evaluation, auxilliaries
from model import Model
import numpy as np

parser = argparse.ArgumentParser(description="Train a model on a given dataset.")
parser.add_argument("--settings", help="Filepath for settings file.", required=True)
parser.add_argument("--dataset", help="Filepath for dataset.", required=True)
args = parser.parse_args()

settings = settings_reader.read(args.settings)
print(settings)



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

entities = io.read_dictionary(entities_path)
relations = io.read_dictionary(relations_path)

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
encoder = model_builder.build_encoder(encoder_settings, train_triplets)
model = model_builder.build_decoder(encoder, decoder_settings)

'''
Construct the optimizer with validation MRR as early stopping metric:
'''

opp = optimizer_parameter_parser.Parser(optimizer_settings)
#opp.set_save_function(model.save) DISABLED SAVING

scorer = evaluation.Scorer(evaluation_settings)
scorer.register_data(train_triplets)
scorer.register_data(valid_triplets)
scorer.register_data(test_triplets)
scorer.register_degrees(train_triplets)
scorer.register_model(model)
scorer.finalize_frequency_computation(train_triplets + valid_triplets + test_triplets)

def score_validation_data(validation_data):
    model.set_variable("GraphSplitSize", len(train_triplets))
    score_summary = scorer.compute_scores(validation_data, verbose=False).get_summary()
    score_summary.dump_degrees('dumps/degrees.in', 'dumps/degrees.out')
    score_summary.dump_frequencies('dumps/near.freq', 'dumps/target.freq')
    #score_summary.pretty_print()

    '''
    f = open('dumps/forward.rels', 'w')
    for row in model.session.run(model.next_component.next_component.C_forward):
        print('\t'.join([str(x) for x in row]), file=f)

    f = open('dumps/backward.rels', 'w')
    for row in model.session.run(model.next_component.next_component.C_backward):
        print('\t'.join([str(x) for x in row]), file=f)

    '''

    model.set_variable("GraphSplitSize", int(general_settings['GraphSplitSize']))

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

if 'NegativeSampleRate' in general_settings:
    ns = auxilliaries.NegativeSampler(int(general_settings['NegativeSampleRate']), general_settings['EntityCount'])
    ns.set_known_positives(train_triplets)

    def t_func(x): #horrible hack!!!
        arr = np.array(x)
        if not encoder.needs_graph():
            return ns.transform(arr)
        else:
            split_size = int(general_settings['GraphSplitSize'])
            graph_split_ids = np.random.choice(len(train_triplets), size=split_size, replace=False)

            graph_split = np.array(train_triplets)[graph_split_ids]
            #gradient_split = np.delete(train_triplets, graph_split_ids, axis=0)

            t = ns.transform(arr)
            return (graph_split, t[0], t[1])

    opp.set_sample_transform_function(t_func)

optimizer_parameters = opp.get_parametrization()

'''
Initialize for training:
'''

graph = np.array(train_triplets)

# Hack for validation evaluation:
model.preprocess(graph)

model.initialize_train()

optimizer_weights = model.get_weights()
optimizer_input = model.get_train_input_variables()
loss = model.get_loss(mode='train') + model.get_regularization()
print(optimizer_input)

'''
Train with Converge:
'''

model.session = tf.Session()
optimizer = build_tensorflow(loss, optimizer_weights, optimizer_parameters, optimizer_input)
optimizer.set_session(model.session)

optimizer.fit(train_triplets, validation_data=valid_triplets)
