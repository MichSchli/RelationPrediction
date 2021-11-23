import imp
import argparse

def generate_sets(triplet_file):
    io = imp.load_source('io', 'code/common/io.py')

    triplet_list = io.read_triplets(triplet_file)

    entity_set = set([])
    relation_set = set([])

    for triplet in triplet_list:
        entity_set.add(triplet[0])
        relation_set.add(triplet[1])
        entity_set.add(triplet[2])

    return entity_set, relation_set

# Generate dictionary from triplet files according to specification:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a dictionary file from a list of triplet files.")
    parser.add_argument("--files", help="Triplet filepaths (separated by #)", required=True)
    parser.add_argument("--relation_dict", help="Filepath for generated relation dictionary", required=True)
    parser.add_argument("--entity_dict", help="Filepath for generated entity dictionary", required=True)

    args = parser.parse_args()

    filepaths = args.files.split('#')

    entities = set([])
    relations = set([])

    for f in filepaths:
        e,r = generate_sets(f)
        entities = entities.union(e)
        relations = relations.union(r)

    entity_file = open(args.entity_dict, 'w+')
    for i,e in enumerate(entities):
        print(str(i)+'\t'+e, file=entity_file)

    relation_file = open(args.relation_dict, 'w+')
    for i,r in enumerate(relations):
        print(str(i)+'\t'+r, file=relation_file)

