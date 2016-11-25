import tensorflow as tf
import imp
import numpy as np

abstract = imp.load_source('abstract_decoder', 'code/experts/decoders/abstract_decoder.py')

class Decoder(abstract.Decoder):

    def transform(self, entity_code):
        return tf.squeeze(tf.matmul(entity_code, self.W1)) + self.b
        
    def decode(self, code):
        energies = tf.reduce_sum(self.transform(code[0])*code[1]*self.transform(code[2]), 1)
        return tf.nn.sigmoid(energies)

    def loss(self, code):
        energies = tf.reduce_sum(self.transform(code[0])*code[1]*self.transform(code[2]), 1)
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(energies, self.Y))

    def fast_decode_all_subjects(self, code, all_subject_codes):
        all_energies = tf.transpose(tf.matmul(self.transform(all_subject_codes), tf.transpose(code[1]*self.transform(code[2]))))
        return tf.nn.sigmoid(all_energies)
    
    def fast_decode_all_objects(self, code, all_object_codes):
        all_energies = tf.matmul(self.transform(code[0])*code[1], tf.transpose(self.transform(all_object_codes)))
        return tf.nn.sigmoid(all_energies)    
    

    def get_weights(self):
        return [self.W1, self.b]
    
    def assign_weights(self, weights):
        self.W1 = tf.Variable(weights[0])
        self.b = tf.Variable(weights[1])

    def initialize_train(self):
        self.W1 = tf.Variable(np.random.randn(200, 200).astype(np.float32))
        self.b = tf.Variable(np.random.randn(200).astype(np.float32))
        self.Y = tf.placeholder(tf.float32, shape=[None])

