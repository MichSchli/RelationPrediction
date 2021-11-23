import tensorflow as tf
from model import Model


class BilinearDiag(Model):
    X = None
    Y = None

    encoder_cache = {'train': None, 'test': None}

    def parse_settings(self):
        self.regularization_parameter = float(self.settings['RegularizationParameter'])

    def compute_codes(self, mode='train'):
        if self.encoder_cache[mode] is not None:
            return self.encoder_cache[mode]

        subject_codes, relation_codes, object_codes = self.next_component.get_all_codes(mode=mode)
        e1s = tf.nn.embedding_lookup(params=subject_codes, ids=self.X[:, 0])
        rs = tf.nn.embedding_lookup(params=relation_codes, ids=self.X[:, 1])
        e2s = tf.nn.embedding_lookup(params=object_codes, ids=self.X[:, 2])

        self.encoder_cache[mode] = (e1s, rs, e2s)
        return self.encoder_cache[mode]


    def get_loss(self, mode='train'):
        e1s, rs, e2s = self.compute_codes(mode=mode)

        energies = tf.reduce_sum(input_tensor=e1s * rs * e2s, axis=1)

        weight = int(self.settings['NegativeSampleRate'])
        weight = 1
        return tf.reduce_mean(input_tensor=tf.nn.weighted_cross_entropy_with_logits(self.Y, energies, weight))

    def local_initialize_train(self):
        self.Y = tf.compat.v1.placeholder(tf.float32, shape=[None])
        self.X = tf.compat.v1.placeholder(tf.int32, shape=[None, 3])

    def local_get_train_input_variables(self):
        return [self.X, self.Y]

    def local_get_test_input_variables(self):
        return [self.X]

    def predict(self):
        e1s, rs, e2s = self.compute_codes(mode='test')
        energies = tf.reduce_sum(input_tensor=e1s * rs * e2s, axis=1)
        return tf.nn.sigmoid(energies)

    def predict_all_subject_scores(self):
        e1s, rs, e2s = self.compute_codes(mode='test')
        all_subject_codes = self.next_component.get_all_subject_codes(mode='test')
        all_energies = tf.transpose(a=tf.matmul(all_subject_codes, tf.transpose(a=rs * e2s)))
        return tf.nn.sigmoid(all_energies)

    def predict_all_object_scores(self):
        e1s, rs, e2s = self.compute_codes(mode='test')
        all_object_codes = self.next_component.get_all_object_codes(mode='test')
        all_energies = tf.matmul(e1s * rs, tf.transpose(a=all_object_codes))
        return tf.nn.sigmoid(all_energies)

    def local_get_regularization(self):
        e1s, rs, e2s = self.compute_codes(mode='train')
        regularization = tf.reduce_mean(input_tensor=tf.square(e1s))
        regularization += tf.reduce_mean(input_tensor=tf.square(rs))
        regularization += tf.reduce_mean(input_tensor=tf.square(e2s))

        return self.regularization_parameter * regularization
