import imp
import random
import numpy as np
import os
import argparse

io = imp.load_source('io', 'code/common/io.py')


parser = argparse.ArgumentParser(description="Make a dataset.")
parser.add_argument("--folder", help="Folder for dataset.", required=True)
args = parser.parse_args()


source_triplets = list(io.read_triplets('data/FB-Toutanova/train.txt'))
source_triplets_valid = list(io.read_triplets('data/FB-Toutanova/valid.txt'))
source_triplets_test = list(io.read_triplets('data/FB-Toutanova/test.txt'))

source_entities = io.read_dictionary('data/FB15k/entities.dict')
reversed_entities = {v: k for k, v in source_entities.items()}


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

data_path = 'data'
folder_path = os.path.join(data_path, args.folder)

ensure_dir(folder_path)

train_path = os.path.join(folder_path, 'train.txt')
valid_path = os.path.join(folder_path, 'valid.txt')
test_path = os.path.join(folder_path, 'test.txt')

def shrink_graph(triplets, target_entities, target_edges, n_target_edges):
    if target_edges.shape[0] > n_target_edges:
        return triplets[target_edges]
    
    print(target_edges.shape[0])

    entity = random.choice(target_entities)

    target_entities = target_entities[target_entities != entity]

    target_edges_sub = np.where(triplets[:,0] == entity)
    target_edges_obj = np.where(triplets[:,2] == entity)

    new_objs = triplets[target_edges_sub][:,2]
    new_subs = triplets[target_edges_obj][:,0]

    new_target_entities = np.unique(np.concatenate((target_entities, new_subs, new_objs)))

    if target_edges.shape[0] == 0:
        new_target_edges = np.concatenate((target_edges_sub[0], target_edges_obj[0]))
    else:
        new_target_edges = np.concatenate((target_edges, target_edges_sub[0], target_edges_obj[0]))

    new_target_edges = np.unique(new_target_edges)

    return shrink_graph(triplets, new_target_entities, new_target_edges, n_target_edges)


#subgraph = shrink_graph(np.array(source_triplets), np.array([random.choice(list(reversed_entities.keys()))]), np.array([]), 40000)
print("Subgraph isolated.")

def split_entities(source_triplets, entities, max_edges=20000):
    d = {k: [] for k in entities}
    for i, triplet in enumerate(source_triplets):
        e1 = triplet[0]
        e2 = triplet[2]

        d[e1].append(i)
        if e1 != e2:
            d[e2].append(i)
    source_array = np.array(source_triplets)
    new_dataset_edges = None

    es = list(d.keys())
    edges_to_remove = []
    while new_dataset_edges is None or len(new_dataset_edges) < max_edges:
        entity = random.choice(es)
        es.remove(entity)
        edges = d[entity]

        if len(edges) == 0:
            continue

        #source_edges = source_array[edges]

        if new_dataset_edges is None:
            new_dataset_edges = edges
        else:
            new_dataset_edges = np.unique(np.concatenate((new_dataset_edges, edges)))

    remaining_old_edges = np.delete(source_array, new_dataset_edges, 0)
    new_dataset_edges = source_array[new_dataset_edges]

    return remaining_old_edges, new_dataset_edges



#rem, train_edges = split_entities(subgraph, reversed_entities, max_edges=20000)
#rem, valid_edges = split_entities(rem, reversed_entities, max_edges=5000)
#rem, test_edges = split_entities(rem, reversed_entities, max_edges=5000)

train_edges, valid_edges = split_entities(source_triplets, reversed_entities, max_edges=10000)
train_edges, test_edges = split_entities(train_edges, reversed_entities, max_edges=10000)

print(train_edges.shape)
print(valid_edges.shape)
print(test_edges.shape)


out_train = open(train_path, 'w+')
for line in train_edges:
    out_train.write('\t'.join(line)+'\n')

out_valid = open(valid_path, 'w+')
for line in valid_edges:
    out_valid.write('\t'.join(line)+'\n')

out_test = open(test_path, 'w+')
for line in test_edges:
    out_test.write('\t'.join(line)+'\n')
