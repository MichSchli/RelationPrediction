import numpy as np

# Dictionary I/O:

def read_dictionary(filename, id_lookup=True):
    d = {}
    for line in open(filename, 'r+'):
        line = line.strip().split('\t')

        if id_lookup:
            d[int(line[0])] = line[1]
        else:
            d[line[1]] = int(line[0])
            
    return d

# Triplet file I/O:

def read_triplets(filename):
    for line in open(filename, 'r+'):
        processed_line = line.strip().split('\t')
        yield processed_line

def read_triplet_file(filename):
    return list(read_triplets(filename))

def read_triplets_as_list(filename, entity_dict, relation_dict):
    entity_dict = read_dictionary(entity_dict, id_lookup=False)
    relation_dict = read_dictionary(relation_dict, id_lookup=False)

    l = []
    for triplet in read_triplets(filename):
        entity_1 = entity_dict[triplet[0]]
        relation = relation_dict[triplet[1]]
        entity_2 = entity_dict[triplet[2]]

        l.append([entity_1, relation, entity_2])

    return l


#print(read_triplets_as_list('data/FB15k/freebase_mtr100_mte100-train.txt', 'data/FB15k/entities.dict', 'data/FB15k/relations.dict'))
