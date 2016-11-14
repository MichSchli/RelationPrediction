import theano
from theano import tensor as T
import imp

abstract = imp.load_source('abstract_decoder', 'code/experts/decoders/abstract_decoder.py')

class Decoder(abstract.Decoder):

    def initialize_train(self):
        self.Y = T.vector('Ys', dtype='float32')

    def decode(self, code):
        energies = T.sum(code[0]*code[1]*code[2], axis=1)
        return T.nnet.sigmoid(energies)

    def loss(self, code):
        prediction = self.decode(code)
        return T.nnet.binary_crossentropy(prediction, self.Y).mean()
    
    def fast_decode_all_subjects(self, code, all_subject_codes):
        all_energies = T.transpose(T.dot(all_subject_codes, T.transpose(code[1]*code[2])))
        return T.nnet.sigmoid(all_energies)
    
    def fast_decode_all_objects(self, code, all_object_codes):
        all_energies = T.dot(code[0]*code[1], T.transpose(all_object_codes))
        return T.nnet.sigmoid(all_energies)    
    
