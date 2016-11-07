import numpy as np

class NegativeSampler():

    negative_sample_rate = None
    n_entities = None
    
    def __init__(self, negative_sample_rate, n_entities):
        self.negative_sample_rate = negative_sample_rate
        self.n_entities = n_entities

    def transform(self, triplets):
        size_of_batch = len(triplets)
        number_to_generate = size_of_batch*self.negative_sample_rate
        
        new_labels = np.zeros((size_of_batch * (self.negative_sample_rate + 1))).astype(np.float32)
        new_indexes = np.tile(triplets, (self.negative_sample_rate + 1,1)).astype(np.int32)
        new_labels[:size_of_batch] = 1

        choices = np.random.binomial(1, 0.5, number_to_generate)
        values = np.random.randint(self.n_entities, size=number_to_generate)

        for i in range(size_of_batch):
            for j in range(self.negative_sample_rate):
                index = i+j*size_of_batch

                if choices[index]:
                    new_indexes[index+size_of_batch,2] = values[index]
                else:
                    new_indexes[index+size_of_batch,0] = values[index]

        return new_indexes, new_labels
