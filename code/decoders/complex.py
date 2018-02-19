import tensorflow as tf
from model import Model


class Complex(Model):
    X = None
    Y = None

    encoder_cache = {'train': None, 'test': None}

    def __init__(self, dimension, settings, next_component=None):
        Model.__init__(self, next_component, settings)
        self.dimension = dimension

    def parse_settings(self):
        self.regularization_parameter = float(self.settings['RegularizationParameter'])

    def compute_codes(self, mode='train'):
        if self.encoder_cache[mode] is not None:
            return self.encoder_cache[mode]

        subject_codes, relation_codes, object_codes = self.next_component.get_all_codes(mode=mode)
        e1s = tf.nn.embedding_lookup(subject_codes, self.X[:, 0])
        rs = tf.nn.embedding_lookup(relation_codes, self.X[:, 1])
        e2s = tf.nn.embedding_lookup(object_codes, self.X[:, 2])

        self.encoder_cache[mode] = (e1s, rs, e2s)
        return self.encoder_cache[mode]


    def get_loss(self, mode='train'):
        e1s, rs, e2s = self.compute_codes(mode=mode)

        e1s_r, e1s_i = self.extract_real_and_imaginary(e1s)
        e2s_r, e2s_i = self.extract_real_and_imaginary(e2s)
        rs_r, rs_i = self.extract_real_and_imaginary(rs)

        energies = tf.reduce_sum(e1s_r * rs_r * e2s_r, 1) \
                   + tf.reduce_sum(e1s_i * rs_r * e2s_i, 1) \
                   + tf.reduce_sum(e1s_r * rs_i * e2s_i, 1) \
                   - tf.reduce_sum(e1s_i * rs_i * e2s_r, 1)

        weight = int(self.settings['NegativeSampleRate'])
        weight = 1
        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.Y, energies, weight))

    def local_initialize_train(self):
        self.Y = tf.placeholder(tf.float32, shape=[None])
        self.X = tf.placeholder(tf.int32, shape=[None, 3])

    def local_get_train_input_variables(self):
        return [self.X, self.Y]

    def local_get_test_input_variables(self):
        return [self.X]

    def predict(self):
        e1s, rs, e2s = self.compute_codes(mode='test')

        e1s_r, e1s_i = self.extract_real_and_imaginary(e1s)
        e2s_r, e2s_i = self.extract_real_and_imaginary(e2s)
        rs_r, rs_i = self.extract_real_and_imaginary(rs)

        energies = tf.reduce_sum(e1s_r * rs_r * e2s_r, 1) \
                   + tf.reduce_sum(e1s_i * rs_r * e2s_i, 1) \
                   + tf.reduce_sum(e1s_r * rs_i * e2s_i, 1) \
                   - tf.reduce_sum(e1s_i * rs_i * e2s_r, 1)

        return tf.nn.sigmoid(energies)

    def extract_real_and_imaginary(self, composite_vector):
        embedding_dim = int(self.dimension/2)
        r = tf.slice(composite_vector, [0, 0], [-1, embedding_dim])
        i = tf.slice(composite_vector, [0, embedding_dim], [-1, embedding_dim])
        return r, i

    def predict_all_subject_scores(self):
        e1s, rs, e2s = self.compute_codes(mode='test')
        all_subject_codes = self.next_component.get_all_subject_codes(mode='test')

        e1s_r, e1s_i = self.extract_real_and_imaginary(all_subject_codes)
        e2s_r, e2s_i = self.extract_real_and_imaginary(e2s)
        rs_r, rs_i = self.extract_real_and_imaginary(rs)

        all_energies = tf.matmul(e1s_r, tf.transpose(rs_r * e2s_r)) \
                       + tf.matmul(e1s_i, tf.transpose(rs_r * e2s_i)) \
                       + tf.matmul(e1s_r, tf.transpose(rs_i * e2s_i)) \
                       - tf.matmul(e1s_i, tf.transpose(rs_i * e2s_r))

        all_energies = tf.transpose(all_energies)
        return tf.nn.sigmoid(all_energies)

    def predict_all_object_scores(self):
        e1s, rs, e2s = self.compute_codes(mode='test')
        all_object_codes = self.next_component.get_all_object_codes(mode='test')

        e1s_r, e1s_i = self.extract_real_and_imaginary(e1s)
        e2s_r, e2s_i = self.extract_real_and_imaginary(all_object_codes)
        rs_r, rs_i = self.extract_real_and_imaginary(rs)

        all_energies = tf.matmul(e1s_r * rs_r, tf.transpose(e2s_r)) \
                   + tf.matmul(e1s_i * rs_r, tf.transpose(e2s_i)) \
                   + tf.matmul(e1s_r * rs_i, tf.transpose(e2s_i)) \
                   - tf.matmul(e1s_i * rs_i, tf.transpose(e2s_r))

        return tf.nn.sigmoid(all_energies)

    def local_get_regularization(self):
        e1s, rs, e2s = self.compute_codes(mode='train')
        regularization = tf.reduce_mean(tf.square(e1s))
        regularization += tf.reduce_mean(tf.square(rs))
        regularization += tf.reduce_mean(tf.square(e2s))

        return self.regularization_parameter * regularization
