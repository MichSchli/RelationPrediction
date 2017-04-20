import numpy as np
import math

class MrrSummary():

    calculate_hits_at = [1,3,10]
    results = {'Raw':{}, 'Filtered':{}}

    
    def __init__(self, raw_ranks, filtered_ranks, in_degrees, out_degrees, vertex_freqs, relation_freqs):
        self.results['Raw'][self.mrr_string()] = self.get_mrr(raw_ranks)
        self.results['Filtered'][self.mrr_string()] = self.get_mrr(filtered_ranks)

        for h in self.calculate_hits_at:
            self.results['Raw'][self.hits_string(h)] = self.get_hits_at_n(raw_ranks,h)
            self.results['Filtered'][self.hits_string(h)] = self.get_hits_at_n(filtered_ranks,h)

        self.results['Raw'][self.degree_string()] = self.get_individual_degree_scores(raw_ranks, in_degrees, out_degrees)
        self.results['Filtered'][self.degree_string()] = self.get_individual_degree_scores(filtered_ranks, in_degrees, out_degrees)

        self.results['Raw'][self.freq_string()] = self.get_individual_freq_scores(raw_ranks, vertex_freqs, relation_freqs)
        self.results['Filtered'][self.freq_string()] = self.get_individual_freq_scores(filtered_ranks, vertex_freqs, relation_freqs)

    def get_individual_freq_scores(self, ranks, vertex_freqs, relation_freqs):
        mrrs = [1/r for r in ranks]
        return zip(mrrs, vertex_freqs, relation_freqs)

    def get_individual_degree_scores(self, ranks, in_degrees, out_degrees):
        in_res = [0] * len(in_degrees)
        out_res = [0] * len(out_degrees)

        for i in range(len(in_res)):
            in_res[i] = (in_degrees[i], 1/ranks[i])
            out_res[i] = (out_degrees[i], 1/ranks[i])

        return in_res, out_res

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

    def freq_string(self):
        return "Frequency"


    def dump_frequencies(self, vertex_filename, relation_filename, filter='Filtered'):

        v_file = open(vertex_filename, 'w+')
        r_file = open(relation_filename, 'w+')

        for mrr, v_freq, r_freq in self.results[filter][self.freq_string()]:
            v_file.write('\t'.join([str(mrr), str(v_freq)]) + '\n')
            r_file.write('\t'.join([str(mrr), str(r_freq)]) + '\n')

        v_file.close()
        r_file.close()


class MrrScore():

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
        self.vertex_freq = [None]*len(dataset)*2
        self.relation_freq = [None]*len(dataset)*2

    def append_line(self, evaluations, gold_idx, filter_idxs, in_degree, out_degree, vertex_freq, relation_freq):
        score_gold = evaluations[gold_idx]
        self.predicted_probabilities[self.pointer] = score_gold
        self.raw_ranks[self.pointer] = np.sum(evaluations >= score_gold)
        self.filtered_ranks[self.pointer] = np.sum(evaluations >= score_gold) - (np.sum(evaluations[filter_idxs] >= score_gold)) + 1
        self.in_degree[self.pointer] = in_degree
        self.out_degree[self.pointer] = out_degree

        self.vertex_freq[self.pointer] = vertex_freq
        self.relation_freq[self.pointer] = relation_freq

        self.pointer += 1
        
    def print_to_file(self, filename):
        outfile = open(filename, 'w+')

        for raw, filtered, prob in zip(self.raw_ranks, self.filtered_ranks, self.predicted_probabilities):
            print(str(raw) +
                  '\t'+ str(filtered) +
                  '\t'+ str(prob),
                  file=outfile)

    def get_summary(self):
        return MrrSummary(self.raw_ranks, self.filtered_ranks, self.in_degree, self.out_degree, self.vertex_freq, self.relation_freq)

    def summarize(self):
        summary = self.get_summary()
        summary.pretty_print()


class AccuracySummary():

    results = {'Filtered':{}, 'Raw':{}}

    def __init__(self, predictions):
        self.results['Filtered'][self.accuracy_string()] = np.mean(predictions)

    def dump_degrees(self, in_file, out_file):
        pass

    def accuracy_string(self):
        return 'Accuracy'

    def pretty_print(self):
        items = [self.accuracy_string()]

        for item in items:
            print(item, end='\t')
            print(str(round(self.results['Filtered'][item],3)), end='\n')


class AccuracyScore():

    def append_all(self, evaluations):
        self.predictions = evaluations

    def summarize(self):
        summary = self.get_summary()
        summary.pretty_print()

    def get_summary(self):
        return AccuracySummary(self.predictions)

        
class Scorer():

    known_subject_triples = {}
    known_object_triples = {}

    in_degree = {}
    out_degree = {}

    relation_freqs = {}
    avg_freq = {}

    def __init__(self, settings):
        self.known_object_triples = {}
        self.known_subject_triples = {}
        self.in_degree = {}
        self.out_degree = {}
        self.relation_freqs = {}
        self.avg_freq = {}
        self.settings = settings

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
            rel = triplet[1]
            obj = triplet[2]

            if sub not in self.in_degree:
                self.in_degree[sub] = 0

            if sub not in self.out_degree:
                self.out_degree[sub] = 0

            if obj not in self.in_degree:
                self.in_degree[obj] = 0

            if obj not in self.out_degree:
                self.out_degree[obj] = 0

            if rel not in self.relation_freqs:
                self.relation_freqs[rel] = 1
            else:
                self.relation_freqs[rel] += 1

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

    def finalize_frequency_computation(self, triples):
        counts = {}
        for triplet in triples:
            sub = triplet[0]
            rel = triplet[1]
            obj = triplet[2]

            if sub not in self.avg_freq:
                self.avg_freq[sub] = 0
                counts[sub] = 0

            if obj not in self.avg_freq:
                self.avg_freq[obj] = 0
                counts[obj] = 0

            self.avg_freq[sub] += self.relation_freqs[rel]
            self.avg_freq[obj] += self.relation_freqs[rel]

            counts[sub] += 1
            counts[obj] += 1

        for k in counts:
            self.avg_freq[k] /= float(counts[k])


    def get_degrees(self, vertex):
        return self.in_degree[vertex], self.out_degree[vertex]

    def compute_accuracy_scores(self, triples, verbose=False):
        score = AccuracyScore()

        if verbose:
            print("Evaluating accuracies...")

        score_vector = self.model.score(triples)
        positives = score_vector[::2]
        negatives = score_vector[1::2]

        evals = positives > negatives

        score.append_all(evals)

        return score

    def compute_scores(self, triples, verbose=False):
        if self.settings['Metric'] == 'MRR':
            return self.compute_mrr_scores(triples, verbose=verbose)
        elif self.settings['Metric'] == 'Accuracy':
            return self.compute_accuracy_scores(triples, verbose=verbose)


    def compute_mrr_scores(self, triples, verbose=False):
        score = MrrScore(triples)

        chunk_size = 1000
        n_chunks = math.ceil(len(triples) / chunk_size)

        for chunk in range(n_chunks):
            i_b = chunk * chunk_size
            i_e = i_b + chunk_size

            triple_chunk = triples[i_b:i_e]
            self.evaluate_mrr(score, triple_chunk, verbose)

        return score

    def evaluate_mrr(self, score, triples, verbose):
        if verbose:
            print("Evaluating subjects...")
            i = 1

        pred_s = self.model.score_all_subjects(triples)
        for evaluations, triplet in zip(pred_s, triples):
            if verbose:
                print("Computing ranks: " + str(i) + " of " + str(len(triples)), end='\r')
                i += 1

            degrees = self.get_degrees(triplet[2])

            avg_freq = self.avg_freq[triplet[2]]
            rel_freq = self.relation_freqs[triplet[1]]

            known_subject_idxs = self.known_subject_triples[(triplet[2], triplet[1])]
            gold_idx = triplet[0]
            score.append_line(evaluations, gold_idx, known_subject_idxs, degrees[0], degrees[1], avg_freq, rel_freq)

        if verbose:
            print("\nEvaluating objects...")
            i = 1

        pred_o = self.model.score_all_objects(triples)
        for evaluations, triplet in zip(pred_o, triples):
            if verbose:
                print("Computing ranks: " + str(i) + " of " + str(len(triples)), end='\r')
                i += 1

            degrees = self.get_degrees(triplet[0])

            avg_freq = self.avg_freq[triplet[0]]
            rel_freq = self.relation_freqs[triplet[1]]

            known_object_idxs = self.known_object_triples[(triplet[0], triplet[1])]
            gold_idx = triplet[2]
            score.append_line(evaluations, gold_idx, known_object_idxs, degrees[0], degrees[1], avg_freq, rel_freq)

        if verbose:
            print("")

    def dump_all_scores(self, triples, subject_file, object_file):
        pred_s = self.model.score_all_subjects(triples)
        f = open(subject_file, 'w')
        for prediction, triplet in zip(pred_s, triples):
            known_subject_idxs = self.known_subject_triples[(triplet[2],triplet[1])]
            target_score = prediction[triplet[0]]
            other_scores = np.delete(prediction, known_subject_idxs)

            print(str(target_score) + " | "+'\t'.join([str(score) for score in other_scores]), file=f)

        pred_o = self.model.score_all_objects(triples)
        f = open(object_file, 'w')
        for prediction, triplet in zip(pred_o, triples):
            known_object_idxs = self.known_object_triples[(triplet[0],triplet[1])]
            target_score = prediction[triplet[2]]
            other_scores = np.delete(prediction, known_object_idxs)

            print(str(target_score) + " | "+'\t'.join([str(score) for score in other_scores]), file=f)



