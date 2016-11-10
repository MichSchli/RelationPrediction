import tensorflow as tf

class Decoder():

    def __init__(self, decoder_settings):
        self.settings = decoder_settings

    def get_additional_ops(self):
        return []
    
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

    def get_regularization(self):
        return 0
