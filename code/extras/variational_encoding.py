import tensorflow as tf
from split_model import SplitModel

class VariationalEncoding(SplitModel):
    vertex_embedding_function = {'train': None, 'test': None}

    def __init__(self, shape, settings, mu_network=None, sigma_network=None):
        SplitModel.__init__(self, [mu_network, sigma_network], settings)

        self.mu_network=mu_network
        self.sigma_network=sigma_network
        self.shape = shape

    def compute_vertex_embeddings(self, mode='train'):
        if self.vertex_embedding_function[mode] is None:
            mu = self.mu_network.get_all_codes(mode=mode)[0]
            log_sigma = self.sigma_network.get_all_codes(mode=mode)[0]
            sigma = tf.exp(log_sigma)

            epsilon = tf.random_normal(self.shape, name='epsilon')
            z = mu + tf.multiply(sigma, epsilon)

            self.vertex_embedding_function[mode] = z

        return self.vertex_embedding_function[mode]

    def local_get_regularization(self):
        mu = self.mu_network.get_all_codes(mode='train')[0]
        log_sigma = self.sigma_network.get_all_codes(mode='train')[0]

        return -0.0005 * tf.reduce_sum(1 + 2*log_sigma - tf.pow(mu, 2) - tf.exp(2*log_sigma))

    def get_all_codes(self, mode='train'):
        collected_messages = self.compute_vertex_embeddings(mode=mode)

        return collected_messages, None, collected_messages

    def get_all_subject_codes(self, mode='train'):
        return self.compute_vertex_embeddings(mode=mode)

    def get_all_object_codes(self, mode='train'):
        return self.compute_vertex_embeddings(mode=mode)