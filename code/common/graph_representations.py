import numpy as np
from scipy.sparse import coo_matrix
import math

class Representation():
    def __init__(self, triples, settings):
        self.triples = np.array(triples)
        self.entity_count = settings['EntityCount']
        self.relation_count = settings['RelationCount']
        self.process(self.triples)

    def get_sender_indices(self):
        return self.sender_indices

    def get_type_indices(self):
        return self.message_types

    def get_receiver_indices(self):
        return self.receiver_indices


    def compute_normalized_values(self, receiver_indices, message_types):
        if self.normalization == "global":
            mrs = receiver_indices
        else:
            mrs = [tuple(x) for x in np.vstack((receiver_indices, message_types)).transpose()]

        counts = {}
        for mr in mrs:
            if mr in counts:
                counts[mr] += 1.0
            else:
                counts[mr] = 1.0

        return np.array([1.0 / counts[mr] for mr in mrs]).astype(np.float32)

    def compute_sparse_mtrs(self, receiver_indices, message_types):
        if self.normalization != "none":
            values = self.compute_normalized_values(receiver_indices, message_types)
        else:
            values = np.ones_like(message_types).astype(np.int32)

        message_indices = np.arange(message_types.shape[0]).astype(np.int32)
        mtr = coo_matrix((values, (receiver_indices, message_indices)), shape=(self.entity_count, self.edge_count),
                                   dtype=np.float32).tocsr()

        return mtr

    def process(self, triplets):
        triplets = triplets.transpose()
        self.sender_indices = np.hstack((triplets[0], triplets[2], np.arange(self.entity_count))).astype(np.int32)
        self.receiver_indices = np.hstack((triplets[2], triplets[0], np.arange(self.entity_count))).astype(np.int32)
        self.message_types = np.hstack(
            (triplets[1] + 1, triplets[1] + self.relation_count + 1, np.zeros(self.entity_count))).astype(np.int32)

