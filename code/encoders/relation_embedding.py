import numpy as np
import tensorflow as tf
from model import Model

class RelationEmbedding(Model):
    embedding_width = None
    W_relation = None

    def parse_settings(self):
        self.embedding_width = int(self.settings['EmbeddingWidth'])

    def local_initialize_train(self):
        relation_initial = np.random.randn(self.relation_count, self.embedding_width).astype(np.float32)

        self.W_relation = tf.Variable(relation_initial)

    def local_get_weights(self):
        return [self.W_relation]

    def get_all_codes(self, mode='train'):
        codes = self.next_component.get_all_codes()
        return codes[0], self.W_relation, codes[2]