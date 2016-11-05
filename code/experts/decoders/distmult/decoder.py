import tensorflow as tf
import imp

abstract = imp.load_source('abstract_decoder', 'code/experts/decoders/abstract_decoder.py')

class Decoder(abstract.Decoder):
    
    def decode(self, code):
        energies = tf.reduce_sum(code[0]*code[1]*code[2], 1)
        return tf.nn.sigmoid(energies)

    def loss(self, code):
        energies = tf.reduce_sum(code[0]*code[1]*code[2], 1)
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(energies, self.Y))

    def fast_decode_all_subjects(self, code, all_subject_codes):
        all_energies = tf.transpose(tf.matmul(all_subject_codes, tf.transpose(code[1]*code[2])))
        return tf.nn.sigmoid(all_energies)
    
    def fast_decode_all_objects(self, code, all_object_codes):
        all_energies = tf.matmul(code[0]*code[1], tf.transpose(all_object_codes))
        return tf.nn.sigmoid(all_energies)    
    
