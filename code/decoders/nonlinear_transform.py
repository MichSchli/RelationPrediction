import tensorflow as tf
from model import Model
import numpy as np


class NonlinearTransform(Model):
    X = None
    Y = None

    encoder_cache = {'train': None, 'test': None}

    def parse_settings(self):
        self.regularization_parameter = float(self.settings['RegularizationParameter'])
        self.dimension = int(self.settings['DecoderDimension'])
        self.embedding_width = int(self.settings['EmbeddingWidth'])

    def local_initialize_train(self):
        transform_matrix_e1 = np.random.normal(0, np.sqrt(1 / (self.embedding_width + self.dimension)),
                                                 size=(self.embedding_width, self.dimension)).astype(np.float32)
        transform_matrix_r = np.random.normal(0, np.sqrt(1 / (self.embedding_width + self.dimension)),
                                                 size=(self.embedding_width, self.dimension)).astype(np.float32)
        transform_matrix_e2 = np.random.normal(0, np.sqrt(1 / (self.embedding_width + self.dimension)),
                                                 size=(self.embedding_width, self.dimension)).astype(np.float32)

        post_transform_matrix = np.random.normal(0, np.sqrt(1/(self.dimension+1)), size=(self.dimension, 1)).astype(np.float32)

        pre_bias_vector = np.zeros(self.dimension).astype(np.float32)

        self.W_e1 = tf.Variable(transform_matrix_e1)
        self.W_r = tf.Variable(transform_matrix_r)
        self.W_e2 = tf.Variable(transform_matrix_e2)

        self.b_pre = tf.Variable(pre_bias_vector)
        self.W_transform = tf.Variable(post_transform_matrix)
        self.b_post = tf.Variable(np.zeros(1).astype(np.float32))

        self.Y = tf.placeholder(tf.float32, shape=[None])
        self.X = tf.placeholder(tf.int32, shape=[None, 3])

    def local_get_weights(self):
        return [self.W_e1, self.W_r, self.W_e2, self.b_pre, self.W_transform, self.b_post]

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

        hidden = tf.matmul(e1s, self.W_e1) + tf.matmul(rs, self.W_r) + tf.matmul(e2s, self.W_e2) + self.b_pre
        activated = tf.nn.relu(hidden)
        output = tf.matmul(activated, self.W_transform) + self.b_post

        energies = tf.squeeze(output)

        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(energies, self.Y))


    def local_get_train_input_variables(self):
        return [self.X, self.Y]

    def local_get_test_input_variables(self):
        return [self.X]

    def predict_all_subject_scores(self):
        print("Warning: Testing broken for this decoder")
        e1s, rs, e2s = self.compute_codes(mode='test')
        all_subject_codes = self.next_component.get_all_subject_codes(mode='test')
        all_energies = tf.transpose(tf.matmul(all_subject_codes, tf.transpose(rs * e2s)))
        return tf.nn.sigmoid(all_energies)

    def predict_all_object_scores(self):
        e1s, rs, e2s = self.compute_codes(mode='test')
        all_object_codes = self.next_component.get_all_object_codes(mode='test')
        all_energies = tf.matmul(e1s * rs, tf.transpose(all_object_codes))
        return tf.nn.sigmoid(all_energies)

    def local_get_regularization(self):
        e1s, rs, e2s = self.compute_codes(mode='train')
        regularization = tf.reduce_mean(tf.square(e1s))
        regularization += tf.reduce_mean(tf.square(rs))
        regularization += tf.reduce_mean(tf.square(e2s))

        return self.regularization_parameter * regularization
