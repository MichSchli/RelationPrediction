import theano
from theano import tensor as T

'''
Defines an abstract model containing methods and variables for use in all experts:
'''
class AbstractModel():

    '''
    Fields:
    '''
    
    n_entities = None
    n_relations = None

    '''
    Initialization methods:
    '''
    
    def __init__(self):
        pass

    def set_entity_count(self, count):
        self.n_entities = count

    def set_relation_count(self, count):
        self.n_relations = count

    '''
    Execution:
    '''

    def wrapper_predict(self, e1_onehot, e2_onehot, relation_id):
        if self.predict_function is None:
            self.predict_function = self.compute_batch_predict_function()

        return self.predict_function(e1_onehot, e2_onehot, relation_id)


    '''
    Computation of theano functions:
    '''
    
    def compute_batch_predict_function(self):
        E1s = T.ivector('E1s')
        E2s = T.ivector('E2s')
        Rs = T.ivector('Rs')
        
        input_variable_list = [E1s, E2s, Rs]
        result = self.theano_batch_predict(E1s, E2s, Rs)

        return theano.function(inputs=input_variable_list, outputs=result)
    
    def compute_batch_loss_function(self):
        input_variable_list = self.get_theano_input_variables()
        loss = self.theano_batch_loss(input_variable_list)
        return theano.function(inputs=input_variable_list, outputs=loss)


