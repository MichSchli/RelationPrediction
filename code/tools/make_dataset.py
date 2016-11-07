import imp

io = imp.load_source('io', 'code/common/io.py')

source_triplets = list(io.read_triplets('data/FB15k/freebase_mtr100_mte100-train.txt'))
source_triplets_valid = io.read_triplets('data/FB15k/freebase_mtr100_mte100-valid.txt')

source_relations = io.read_dictionary('data/FB15k/relations.dict')

reversed_relations = {v: k for k, v in source_relations.items()}
d = {k:0 for k in source_relations}

for triplet in source_triplets:
    i = reversed_relations[triplet[1]]
    d[i] += 1

tuples = sorted(d.items(), key=lambda x: x[1], reverse=True)

limited_relations = [source_relations[t[0]] for t in tuples[:10]]

limited_triplets = [t for t in source_triplets if t[1] in limited_relations]
limited_triplets_valid = [t for t in source_triplets_valid if t[1] in limited_relations]

print(len(limited_triplets))
print(len(limited_triplets_valid))

out_train = open('out_train.txt', 'w+')
for line in limited_triplets:
    out_train.write('\t'.join(line)+'\n')


out_valid = open('out_valid.txt', 'w+')
for line in limited_triplets_valid:
    out_valid.write('\t'.join(line)+'\n')
