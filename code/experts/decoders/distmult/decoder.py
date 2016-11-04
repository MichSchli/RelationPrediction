import tensorflow as tf

class Decoder():

    def __init__(self, decoder_settings):
        self.settings = decoder_settings

    def initialize_train(self):
        self.Y = tf.placeholder(tf.float32, shape=[None])

    def initialize_test(self):
        pass

    def get_gold_input_variable(self):
        return self.Y

    def get_weights(self):
        return []

    def preprocess(self, triplets):
        pass
    
    def assign_weights(self, weights):
        pass
    
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
    
    def get_regularization(self):
        return 0
