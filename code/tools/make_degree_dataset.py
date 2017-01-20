import imp
import random
import numpy as np
import os
import argparse

io = imp.load_source('io', 'code/common/io.py')


parser = argparse.ArgumentParser(description="Make a dataset.")
parser.add_argument("--folder", help="Folder for dataset.", required=True)
args = parser.parse_args()


source_triplets = list(io.read_triplets('data/FB15k/train.txt'))
source_triplets_valid = list(io.read_triplets('data/FB15k/valid.txt'))
source_triplets_test = list(io.read_triplets('data/FB15k/test.txt'))

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

    if target_edges_sub[0].shape[0] + target_edges_obj[0].shape[0] > 500:
        return shrink_graph(triplets, target_entities, target_edges, n_target_edges)

    new_objs = triplets[target_edges_sub][:,2]
    new_subs = triplets[target_edges_obj][:,0]

    new_target_entities = np.unique(np.concatenate((target_entities, new_subs, new_objs)))

    if target_edges.shape[0] == 0:
        new_target_edges = np.concatenate((target_edges_sub[0], target_edges_obj[0]))
    else:
        new_target_edges = np.concatenate((target_edges, target_edges_sub[0], target_edges_obj[0]))

    new_target_edges = np.unique(new_target_edges)

    return shrink_graph(triplets, new_target_entities, new_target_edges, n_target_edges)


subgraph = shrink_graph(np.array(source_triplets), np.array([random.choice(list(reversed_entities.keys()))]), np.array([]), 500)
print("Subgraph isolated.")


#Create dictionary:
d = {}

for edge in subgraph:
    if edge[0] not in d:
        d[edge[0]] = []

    if edge[2] not in d:
        d[edge[2]] = []

    if np.random.binomial(1, 0.8):
        d[edge[0]].append(edge[2])
        #d[edge[2]].append(edge[0])

for k in d:
    d[k] = np.unique(d[k])

d2 = {}

for k in d:
    d2[k] = []
    for e in d[k]:
        d2[k].extend(d[e])

    d2[k] = np.unique(d2[k])

train_edges = []

for k in d2:
    for e in d2[k]:
        train_edges.append([k, '2nd_order_edge', e])

train_edges = np.array(train_edges)

sample = np.random.choice(train_edges.shape[0], size=500, replace=False)
valid_edges = train_edges[sample]
train_edges = np.delete(train_edges, sample, axis=0)


sample = np.random.choice(train_edges.shape[0], size=500, replace=False)
test_edges = train_edges[sample]
train_edges = np.delete(train_edges, sample, axis=0)

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
