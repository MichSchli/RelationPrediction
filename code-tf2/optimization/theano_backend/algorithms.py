from optimization.abstract import IOptimizer
import random
from theano import tensor as T
import theano
import numpy as np
import tensorflow as tf

class GradientDescent(IOptimizer):

    learning_rate = None

    def valid(self):
        return self.learning_rate is not None

    def theano_process_update_function(self, parameters, loss_function):
        gradient = self.compute_gradient_function(parameters, loss_function)

        update_list = self.next_component.theano_process_update_function(parameters, loss_function)
        lower_count = len(update_list)
        
        update_list += [None]*len(gradient)
        for i in range(len(gradient)):
            update_list[lower_count + i] = (parameters[i], parameters[i] - self.learning_rate * gradient[i])

        return update_list


class AdaGrad(IOptimizer):

    learning_rate = None
    epsillon = 1e-8
    historical_gradient = None

    def initialize_running_average(self, parameters):
        self.historical_gradient = [None]*len(parameters)
        
        for i,parameter in enumerate(parameters):
            self.historical_gradient[i] = theano.shared((np.zeros_like(parameter.get_value())).astype(np.float32))

            
    def valid(self):
        return self.learning_rate is not None

    def theano_process_update_function(self, parameters, loss_function):
        gradient = self.compute_gradient_function(parameters, loss_function)
        update_list = self.next_component.theano_process_update_function(parameters, loss_function)
        lower_count = len(update_list)

        self.initialize_running_average(parameters)

        update_list += [None]*(len(gradient)*2)
        for i in range(len(gradient)):
            new_historical_gradient = self.historical_gradient[i] + gradient[i] * gradient[i]
            scaling = T.sqrt(new_historical_gradient + self.epsillon)
            delta = (self.learning_rate / scaling) * gradient[i]
            
            update_list[lower_count + i] = (parameters[i], parameters[i] - delta)            
            update_list[lower_count + i + len(gradient)] = (self.historical_gradient[i], new_historical_gradient)

        return update_list


class RmsProp(IOptimizer):

    learning_rate = None
    historical_weight = None
    epsillon = 1e-8
    historical_gradient = None

    def initialize_running_average(self, parameters):
        self.historical_gradient = [None]*len(parameters)
        
        for i,parameter in enumerate(parameters):
            self.historical_gradient[i] = theano.shared(np.zeros_like(parameter.get_value()).astype(np.float32))

    def valid(self):
        return self.learning_rate is not None and self.historical_weight is not None

    def theano_process_update_function(self, parameters, loss_function):
        gradient = self.compute_gradient_function(parameters, loss_function)
        update_list = self.next_component.theano_process_update_function(parameters, loss_function)
        lower_count = len(update_list)

        self.initialize_running_average(parameters)

        update_list += [None]*(len(gradient)*2)
        for i in range(len(gradient)):
            new_historical_gradient = self.historical_weight*self.historical_gradient[i] + (1-self.historical_weight)*gradient[i] * gradient[i]
            scaling = T.sqrt(new_historical_gradient + self.epsillon)
            delta = (self.learning_rate / scaling) * gradient[i]
            
            update_list[lower_count + i] = (parameters[i], parameters[i] - delta)            
            update_list[lower_count + i + len(gradient)] = (self.historical_gradient[i], new_historical_gradient)

        return update_list


    
class Adam(IOptimizer):

    learning_rate = None
    historical_moment_weight = 0.9
    historical_gradient_weight = 0.999

    epsillon = 1e-8
    historical_gradient = None
    
    def initialize_running_average(self, parameters):
        self.historical_gradient = [None]*len(parameters)
        self.historical_moment = [None]*len(parameters)
        
        for i,parameter in enumerate(parameters):
            self.historical_gradient[i] = theano.shared(np.zeros_like(parameter.get_value()).astype(np.float32))
            self.historical_moment[i] = theano.shared(np.zeros_like(parameter.get_value()).astype(np.float32))

        self.theano_iteration = theano.shared(np.cast['float32'](1))
        
    def valid(self):
        return self.learning_rate is not None and self.historical_gradient_weight is not None and self.historical_moment_weight is not None

    def theano_process_update_function(self, parameters, loss_function):
        gradient = self.compute_gradient_function(parameters, loss_function)
        update_list = self.next_component.theano_process_update_function(parameters, loss_function)
        lower_count = len(update_list)

        self.initialize_running_average(parameters)

        update_list += [None]*(len(gradient)*3+1)
        for i in range(len(gradient)):
            new_historical_moment = self.historical_moment_weight*self.historical_moment[i] + (1-self.historical_moment_weight) * gradient[i]
            new_historical_gradient = self.historical_gradient_weight*self.historical_gradient[i] + (1-self.historical_gradient_weight)*T.sqr(gradient[i])

            corrected_learning_rate = self.learning_rate #* T.sqrt(1 - self.historical_gradient_weight**self.iteration) / (1 - self.historical_moment_weight**self.iteration)
            corrected_moment = new_historical_moment / (1 - self.historical_moment_weight**self.theano_iteration)
            corrected_gradient = new_historical_gradient / (1 - self.historical_gradient_weight**self.theano_iteration)
            
            scaling = T.sqrt(corrected_gradient) + self.epsillon
            delta = (corrected_learning_rate / scaling) * corrected_moment
            
            update_list[lower_count + i] = (parameters[i], parameters[i] - delta)            
            update_list[lower_count + i + len(gradient)] = (self.historical_gradient[i], new_historical_gradient)
            update_list[lower_count + i + len(gradient)*2] = (self.historical_moment[i], new_historical_moment)

        update_list[-1] = (self.theano_iteration, self.theano_iteration + 1)

        return update_list


    
class GradientClipping(IOptimizer):

    max_norm = None

    def valid(self):
        return self.max_norm is not None

    def compute_gradient_function(self, parameters, loss_function):
        gradient = list(self.next_component.compute_gradient_function(parameters, loss_function))

        norm = 0
        for grad in gradient:
            norm += (grad * grad).sum()
        norm = T.sqrt(norm)

        for i,grad in enumerate(gradient):
            gradient[i] = grad * T.minimum(1, self.max_norm / norm)

        return gradient

