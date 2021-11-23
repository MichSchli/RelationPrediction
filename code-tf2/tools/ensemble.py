import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Combine the output of multiple runs in an ensemble.")
parser.add_argument("--p1", help="Filepath for prediction files.", required=True)
parser.add_argument("--p2", help="Filepath for prediction files.", required=True)
parser.add_argument("--method", help="Method used for combining models.", required=True)
args = parser.parse_args()

class CutoffEnsemble():

    def __init__(self, cutoff, model_1, model_2):
        self.cutoff = cutoff
        self.model_1 = model_1
        self.model_2 = model_2

    def read_degree_file(self, filename):
        for line in open(filename, 'r'):
            degree, mrr = line.strip().split('\t')
            degree = int(degree)
            mrr = float(mrr)

            yield degree, mrr

    def combine(self):
        left_in = list(self.read_degree_file(self.model_1 + '/degrees.in'))
        left_out = list(self.read_degree_file(self.model_1 + '/degrees.out'))
        right_in = list(self.read_degree_file(self.model_2 + '/degrees.in'))
        right_out = list(self.read_degree_file(self.model_2 + '/degrees.out'))

        for li, lo, ri, ro in zip(left_in, left_out, right_in, right_out):
            deg = li[0] + lo[0]
            if deg < self.cutoff:
                yield li[1]
                yield lo[1]
            else:
                yield ri[1]
                yield ro[1]

    def combined_mrr(self):
        return np.mean(list(self.combine()))

class WeightEnsemble():

    def __init__(self, weight, model_1, model_2):
        self.weight = weight
        self.model_1 = model_1
        self.model_2 = model_2

    def read_mrr_file(self, filename):
        for line in open(filename, 'r'):
            parts = line.strip().split(' | ')
            target = float(parts[0])
            others = [float(p) for p in parts[1].split('\t')]
            yield target, others

    def combine(self):
        for left, right in zip(self.read_mrr_file(self.model_1+'/subjects.test'),
                               self.read_mrr_file(self.model_2+ '/subjects.test')):
            yield self.combine_prediction(left, right)

        for left, right in zip(self.read_mrr_file(self.model_1+'/objects.test'),
                               self.read_mrr_file(self.model_2+ '/objects.test')):
            yield self.combine_prediction(left, right)

    def combine_prediction(self, left, right):
        target = self.weight * left[0] + (1 - self.weight) * right[0]
        others = [None] * len(left[1])
        for i in range(len(others)):
            others[i] = self.weight * left[1][i] + (1 - self.weight) * right[1][i]
        rank = np.sum(np.array(others) >= target) + 1

        return rank

    def combined_mrr(self):
        return np.mean(1 / self.ranks)

    def compute_ranks(self):
        self.ranks = np.array(list(self.combine()))

    def hits_at(self, threshold):
        hits = self.ranks[self.ranks <= threshold]
        return len(hits) / len(self.ranks)

if args.method == 'cutoff':
    model = CutoffEnsemble(1000, args.p1, args.p2)
elif args.method == 'weighted_sum':
    model = WeightEnsemble(0.5, args.p1, args.p2)

model.compute_ranks()

print(model.combined_mrr())
print(model.hits_at(1))
print(model.hits_at(3))
print(model.hits_at(10))