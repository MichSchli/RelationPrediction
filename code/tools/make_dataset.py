import imp
import random
import numpy as np

io = imp.load_source('io', 'code/common/io.py')

source_triplets = list(io.read_triplets('data/FB15k/train.txt'))
source_triplets_valid = list(io.read_triplets('data/FB15k/valid.txt'))
source_triplets_test = list(io.read_triplets('data/FB15k/test.txt'))

source_entities = io.read_dictionary('data/FB15k/entities.dict')
reversed_entities = {v: k for k, v in source_entities.items()}


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

        source_edges = source_array[edges]

        if new_dataset_edges is None:
            new_dataset_edges = source_edges
        else:
            new_dataset_edges = np.vstack((new_dataset_edges, source_edges))

        edges_to_remove.extend(edges)
        new_dataset_edges = np.vstack({tuple(row) for row in new_dataset_edges})

    remaining_old_edges = np.delete(source_array, edges_to_remove, 0)

    return remaining_old_edges, new_dataset_edges


train_edges, valid_edges = split_entities(source_triplets, reversed_entities)
print(train_edges.shape)
print(valid_edges.shape)

train_edges, test_edges = split_entities(train_edges, reversed_entities)
print(train_edges.shape)
print(test_edges.shape)


out_train = open('out_train.txt', 'w+')
for line in train_edges:
    out_train.write('\t'.join(line)+'\n')

out_valid = open('out_valid.txt', 'w+')
for line in valid_edges:
    out_valid.write('\t'.join(line)+'\n')

out_test = open('out_test.txt', 'w+')
for line in test_edges:
    out_test.write('\t'.join(line)+'\n')