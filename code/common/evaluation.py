import numpy as np

class Summary():

    calculate_hits_at = [1,3,10]
    results = {'Raw':{}, 'Filtered':{}}
    
    def __init__(self, raw_ranks, filtered_ranks, in_degrees, out_degrees):
        self.results['Raw'][self.mrr_string()] = self.get_mrr(raw_ranks)
        self.results['Filtered'][self.mrr_string()] = self.get_mrr(filtered_ranks)

        for h in self.calculate_hits_at:
            self.results['Raw'][self.hits_string(h)] = self.get_hits_at_n(raw_ranks,h)
            self.results['Filtered'][self.hits_string(h)] = self.get_hits_at_n(filtered_ranks,h)

        self.results['Raw'][self.degree_string()] = self.get_degree_scores(raw_ranks, in_degrees, out_degrees)
        self.results['Filtered'][self.degree_string()] = self.get_degree_scores(filtered_ranks, in_degrees, out_degrees)

    def get_degree_scores(self, ranks, in_degrees, out_degrees):
        in_buckets = [0]*max(in_degrees)
        out_buckets = [0]*max(out_degrees)

        in_counts = [0] * max(in_degrees)
        out_counts = [0] * max(out_degrees)

        for in_degree, out_degree,rank in zip(in_degrees, out_degrees, ranks):
            in_buckets[in_degree-1] += 1/rank
            out_buckets[out_degree-1] += 1/rank

            in_counts[in_degree - 1] += 1
            out_counts[out_degree - 1] += 1

        in_res = []
        out_res = []
        for i in range(len(in_buckets)):
            if in_counts[i] > 0:
                in_res.append((i, in_buckets[i] / in_counts[i]))

        for i in range(len(out_buckets)):
            if out_counts[i] > 0:
                out_res.append((i, out_buckets[i] / out_counts[i]))

        return in_res, out_res


    def mrr_string(self):
        return 'MRR'

    def hits_string(self, n):
        return 'H@'+str(n)

    def degree_string(self):
        return "Degree"
            
    def pretty_print(self):
        print('\tRaw\tFiltered')

        items = [self.mrr_string()]
        for h in self.calculate_hits_at:
            items.append(self.hits_string(h))

        for item in items:
            print(item, end='\t')
            print(str(round(self.results['Raw'][item],3)), end='\t')
            print(str(round(self.results['Filtered'][item],3)))
            
    def get_mrr(self, ranks):
        mean_reciprocal_rank = 0.0
        for rank in ranks:
            mean_reciprocal_rank += 1/rank
        return mean_reciprocal_rank / len(ranks)

    def get_hits_at_n(self, ranks, n):
        hits = 0.0
        for rank in ranks:
            if rank <= n:
                hits += 1
        return hits / len(ranks)

    def dump_degrees(self, in_filename, out_filename, filter='Filtered'):

        in_file = open(in_filename, 'w+')
        for i,deg in self.results[filter][self.degree_string()][0]:
            in_file.write('\t'.join([str(i+1), str(deg)]) + '\n')

        in_file.close()

        out_file = open(out_filename, 'w+')
        for i, deg in self.results[filter][self.degree_string()][1]:
            out_file.write('\t'.join([str(i+1), str(deg)]) + '\n')

        out_file.close()


class Score():

    raw_ranks = []
    filtered_ranks = []
    predicted_probabilities = []
    in_degree = []
    out_degree = []
    pointer = 0
    
    def __init__(self, dataset):
        self.raw_ranks = [None]*len(dataset)*2
        self.filtered_ranks = [None]*len(dataset)*2
        self.predicted_probabilities = [None]*len(dataset)*2
        self.in_degree = [None]*len(dataset)*2
        self.out_degree = [None]*len(dataset)*2

    def append_line(self, evaluations, gold_idx, filter_idxs, in_degree, out_degree):
        score_gold = evaluations[gold_idx]
        self.predicted_probabilities[self.pointer] = score_gold
        self.raw_ranks[self.pointer] = np.sum(evaluations >= score_gold)
        self.filtered_ranks[self.pointer] = np.sum(evaluations >= score_gold) - (np.sum(evaluations[filter_idxs] >= score_gold)) + 1
        self.in_degree[self.pointer] = in_degree
        self.out_degree[self.pointer] = out_degree

        self.pointer += 1
        
    def print_to_file(self, filename):
        outfile = open(filename, 'w+')

        for raw, filtered, prob in zip(self.raw_ranks, self.filtered_ranks, self.predicted_probabilities):
            print(str(raw) +
                  '\t'+ str(filtered) +
                  '\t'+ str(prob),
                  file=outfile)

    def get_summary(self):
        return Summary(self.raw_ranks, self.filtered_ranks, self.in_degree, self.out_degree)

    def summarize(self):
        summary = self.get_summary()
        summary.pretty_print()
        
        
class Scorer():

    known_subject_triples = {}
    known_object_triples = {}

    in_degree = {}
    out_degree = {}

    def __init__(self):
        self.known_object_triples = {}
        self.known_subject_triples = {}
        self.in_degree = {}
        self.out_degree = {}

    def extend_triple_dict(self, dictionary, triplets, object_list=True):
        for triplet in triplets:
            if object_list:
                key = (triplet[0], triplet[1])
                value = triplet[2]
            else:
                key = (triplet[2],triplet[1])
                value = triplet[0]
        
            if key not in dictionary:
                dictionary[key] = [value]
            elif value not in dictionary[key]:
                dictionary[key].append(value)

    def register_data(self, triples):
        for triplet in triples:
            sub = triplet[0]
            obj = triplet[2]

            if sub not in self.in_degree:
                self.in_degree[sub] = 0

            if sub not in self.out_degree:
                self.out_degree[sub] = 0

            if obj not in self.in_degree:
                self.in_degree[obj] = 0

            if obj not in self.out_degree:
                self.out_degree[obj] = 0

        self.extend_triple_dict(self.known_subject_triples, triples, object_list=False)
        self.extend_triple_dict(self.known_object_triples, triples)

    def register_degrees(self, triples):
        for triplet in triples:
            sub = triplet[0]
            obj = triplet[2]

            self.in_degree[obj] += 1
            self.out_degree[sub] += 1

    def register_model(self, model):
        self.model = model

    def get_degrees(self, vertex):
        return self.in_degree[vertex], self.out_degree[vertex]

    def compute_scores(self, triples, verbose=False):
        score = Score(triples)

        if verbose:
            print("Evaluating subjects...")
            i = 1
            
        pred_s = self.model.score_all_subjects(triples)

        for evaluations, triplet in zip(pred_s, triples):
            if verbose:
                print("Computing ranks: "+str(i)+" of "+str(len(triples)), end='\r')
                i += 1

            degrees = self.get_degrees(triplet[2])
            known_subject_idxs = self.known_subject_triples[(triplet[2],triplet[1])]
            gold_idx = triplet[0]            
            score.append_line(evaluations, gold_idx, known_subject_idxs, degrees[0], degrees[1])

        if verbose:
            print("\nEvaluating objects...")
            i = 1
            
        pred_o = self.model.score_all_objects(triples)

        for evaluations, triplet in zip(pred_o, triples):
            if verbose:
                print("Computing ranks: "+str(i)+" of "+str(len(triples)), end='\r')
                i += 1

            degrees = self.get_degrees(triplet[0])
            known_object_idxs = self.known_object_triples[(triplet[0],triplet[1])]
            gold_idx = triplet[2]
            score.append_line(evaluations, gold_idx, known_object_idxs, degrees[0], degrees[1])

        print("")
        return score
