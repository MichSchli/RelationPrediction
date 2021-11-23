import numpy as np
import tensorflow as tf
from model import Model

class RelationEmbedding(Model):
    shape=None

    def __init__(self, shape, settings, next_component=None):
        Model.__init__(self, next_component, settings)
        self.shape = shape

    def parse_settings(self):
        self.embedding_width = int(self.settings['CodeDimension'])

    def local_initialize_train(self):
        relation_initial = np.random.randn(self.shape[0], self.shape[1]).astype(np.float32)

        self.W_relation = tf.Variable(relation_initial)

    def local_get_weights(self):
        return [self.W_relation]

    def get_all_codes(self, mode='train'):
        codes = self.next_component.get_all_codes(mode=mode)
        return codes[0], self.W_relation, codes[2]