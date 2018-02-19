import theano
from theano import tensor as T
import tensorflow as tf

'''
Optimizer interface:
'''
class IOptimizer():

    next_component = None
    
    def __init__(self, next_component, parameters):
        self.next_component = next_component

        for k,v in parameters.items():
            setattr(self,k,v)

    def verify(self):
        return self.valid() and self.next_component.verify()
            
    def process_loss_function(self, loss_function):
        return self.next_component.process_loss_function(loss_function)

    def theano_process_update_function(self, parameters, loss_function):
        return self.next_component.theano_process_update_function(parameters, loss_function)

    def process_data(self, data):
        return self.next_component.process_data(data)
    
    def compute_gradient_function(self, parameters, loss_function):
        return self.next_component.compute_gradient_function(parameters, loss_function)
    
    def postprocess(self, loss):
        if self.next_component is not None:
            return self.next_component.postprocess(loss)
        else:
            return 'continue'

    def set_iteration(self, iteration):
        self.iteration = iteration

        if self.next_component is not None:
            self.next_component.set_iteration(iteration)

    def next_batch(self):
        return self.next_component.next_batch()

    def get_message(self):
        return self.next_component.get_message()

    def set_validation_data(self, validation_data):
        self.validation_data = validation_data

        if self.next_component is not None:
            self.next_component.set_validation_data(validation_data)
            
    def set_training_data(self, training_data):
        self.training_data = training_data

        if self.next_component is not None:
            self.next_component.set_training_data(training_data)

    '''
    TF:
    '''

    def get_additional_ops(self):
        return self.next_component.get_additional_ops()
    
    def set_session(self, session):
        self.session = session

        if self.next_component is not None:
            self.next_component.set_session(session)
    
    def process_gradient_function(self, loss_function, parameters_to_optimize):
        return self.next_component.process_gradient_function(loss_function, parameters_to_optimize)

    def process_update_function(self, gradient_function, parameters_to_optimize):
        return self.next_component.process_update_function(gradient_function, parameters_to_optimize)

'''
Base optimizer:
'''
class BaseOptimizer(IOptimizer):

    def __init__(self):
        pass

    def verify(self):
        return True

    def get_additional_ops(self):
        return []
    
    def process_loss_function(self, loss_function):
        return loss_function

    def theano_process_update_function(self, parameters, loss_function):
        return []
    
    def process_update_function(self, gradient_function):
        pass

    def compute_gradient_function(self, parameters, loss_function):
        return T.grad(loss_function, wrt=parameters)
    
    def next_batch(self):
        return self.training_data

    def get_message(self):
        return None

    def process_data(self, data):
        return data

    def process_gradient_function(self, loss_function, parameters_to_optimize):
        return tf.gradients(loss_function, parameters_to_optimize)
