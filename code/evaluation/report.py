import argparse

parser = argparse.ArgumentParser(description="bla.")
parser.add_argument("--filepath", help="lal", required=True)
args = parser.parse_args()

mean_reciprocal_rank = [0,0]
hits_at_one = [0,0]
hits_at_three = [0,0]
hits_at_ten = [0,0]

number_of_samples = 0

for line in open(args.filepath):
    line = line.strip().split('\t')

    number_of_samples += 1
    
    mean_reciprocal_rank[0] += 1/(float(line[0]))
    mean_reciprocal_rank[1] += 1/(float(line[1]))

    hits_at_one[0] += 1 if int(line[0]) <= 1 else 0
    hits_at_three[0] += 1 if int(line[0]) <= 3 else 0
    hits_at_ten[0] += 1 if int(line[0]) <= 10 else 0

    hits_at_one[1] += 1 if int(line[1]) <= 1 else 0
    hits_at_three[1] += 1 if int(line[1]) <= 3 else 0
    hits_at_ten[1] += 1 if int(line[1]) <= 10 else 0
    

scores = []
scores.append(mean_reciprocal_rank)
scores.append(hits_at_one)
scores.append(hits_at_three)
scores.append(hits_at_ten)

row_headers = ['MRR', 'h@1', 'h@3', 'h@10']
column_headers = ['Raw', 'Filtered'] 

header_string = '\t '+'\t'.join(column_headers)
print(header_string)
for i,header in enumerate(row_headers):
    print(header + '\t ' + '\t'.join([str(round(x/float(number_of_samples),3)) for x in scores[i]]))

