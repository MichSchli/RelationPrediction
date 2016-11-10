import tensorflow as tf

class Encoder():

    def __init__(self, encoder_settings):
        self.settings = encoder_settings

    def get_additional_ops(self):
        return []
    
    def get_input_variables(self):
        return [self.X]

    def initialize_test(self):
        self.X = tf.placeholder(tf.int32, shape=[None,3])

    def preprocess(self, triplets):
        pass

    def get_regularization(self):
        return 0

    def parameter_count(self):
        return 0

    def assign_weights(self, weights):
        pass

    def get_weights(self):
        return []
