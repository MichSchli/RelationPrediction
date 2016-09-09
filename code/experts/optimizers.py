import theano
from theano import tensor as T
import gc
import numpy as np

class AbstractBatchOptimizer():

    max_iterations = 10
    batch_size = 100

    def __init__(self):
        pass

    def train(self, model, train_triplets, valid_triplets, model_path):
        update_function = self.compute_update_function(model)
        loss_function = model.compute_batch_loss_function()

        for i in range(self.max_iterations):     
            print(i)

            # THIS IS JUST ONE OF FOUR APPROACHES TO GENERATING NEGATIVE VALIDATION SAMPLES!!!
            # Sampling here is right, but train triplets should not be sampled from
            validation_samples = model.process_train_triplets(valid_triplets, valid_triplets + train_triplets)
            
            print(loss_function(*validation_samples))
            for chunk in self.chunk_iterate(train_triplets):
                processed_samples = model.process_train_triplets(chunk, train_triplets)
                train_loss, train_reg = update_function(*processed_samples)

            model.save(model_path)

                
    def chunk_iterate(self, l):
        chunk = [None]*self.batch_size
        iterator = 0
        for element in l:
            chunk[iterator] = element
            iterator += 1
            
            if iterator == self.batch_size:
                iterator = 0
                yield chunk

        if iterator > 0:
            yield chunk[0:iterator]

            
class BatchGradientDescent(AbstractBatchOptimizer):

    learning_rate = 0.1
    regularization_parameter = 0.01
    
    def __init__(self):
        pass

    def update_parameter(self, parameter, gradient):
        return parameter - (self.learning_rate / self.batch_size) * gradient
    
    def compute_update_function(self, model):
        input_variable_list = model.get_theano_input_variables()
        loss,_ = theano.scan(model.theano_loss,
                             sequences=input_variable_list)

        sumloss = T.sum(loss)
        regularization = model.theano_l2_regularization()
        sumloss += self.regularization_parameter * regularization
        
        results = T.grad(sumloss, wrt=model.get_weights())

        update_list = [None]*len(results)
        for i in range(len(results)):
            update_list[i] = (model.get_weights()[i], self.update_parameter(model.get_weights()[i], results[i]))
        
        return theano.function(inputs=input_variable_list, outputs=[sumloss, regularization], updates=update_list)


class AdamOptimizer(AbstractBatchOptimizer):

    learning_rate = 0.01
    regularization_parameter = 0.01

    epsillon = 10**(-8)
    beta_1 = 0.9
    beta_2 = 0.999
    
    def __init__(self):
        pass

    def update_parameter(self, parameter, m, v):
        return parameter - (self.learning_rate / self.batch_size) * m / (T.sqrt(v) + self.epsillon)

    def initialize_moment_estimates(self, parameter_shape_list):
        self.iteration_number = theano.shared(np.cast['float32'](1))
        self.m_previous = [None]*len(parameter_shape_list)
        self.v_previous = [None]*len(parameter_shape_list)
        
        for i,parameter_shape in enumerate(parameter_shape_list):
            self.m_previous[i] = theano.shared(np.zeros(parameter_shape).astype(np.float32))
            self.v_previous[i] = theano.shared(np.zeros(parameter_shape).astype(np.float32))

            
    def compute_update_function(self, model):
        input_variable_list = model.get_theano_input_variables()

        parameter_shapes = model.get_weight_shapes()
        self.initialize_moment_estimates(parameter_shapes)
        
        loss,_ = theano.scan(model.theano_loss,
                             sequences=input_variable_list)

        sumloss = T.sum(loss)
        regularization = model.theano_l2_regularization()
        sumloss += self.regularization_parameter * regularization
        
        gradient = T.grad(sumloss, wrt=model.get_weights())

        m = [None]*len(gradient)
        v = [None]*len(gradient)
        update_list = [None]*(len(gradient)*3+1)
        for i in range(len(gradient)):

            m[i] = self.beta_1 * self.m_previous[i] + (1-self.beta_1) * gradient[i]
            v[i] = self.beta_2 * self.v_previous[i] + (1-self.beta_2) * (gradient[i] * gradient[i])

            m[i] = m[i] / (1 - self.beta_1**self.iteration_number)
            v[i] = v[i] / (1 - self.beta_2**self.iteration_number)
            
            update_list[i] = (model.get_weights()[i], self.update_parameter(model.get_weights()[i], m[i], v[i]))

            update_list[i+len(gradient)] = (self.m_previous[i], m[i])
            update_list[i+len(gradient)*2] = (self.v_previous[i], v[i])

        update_list[-1] = (self.iteration_number, self.iteration_number+1)
        return theano.function(inputs=input_variable_list, outputs=[sumloss, regularization], updates=update_list)

    
class RmsProp(AbstractBatchOptimizer):

    learning_rate = 0.1
    regularization_parameter = 0.01
    epsillon = 10**(-6)
    decay_rate = 0.9
    
    def __init__(self):
        pass

    def update_parameter(self, parameter, gradient, rmsgrad):
        return parameter - (self.learning_rate / (self.batch_size*rmsgrad)) * gradient

    def initialize_running_average(self, parameter_shape_list):
        self.running_average = [None]*len(parameter_shape_list)
        
        for i,parameter_shape in enumerate(parameter_shape_list):
            self.running_average[i] = theano.shared(np.zeros(parameter_shape).astype(np.float32))
            
    
    def compute_update_function(self, model):
        input_variable_list = model.get_theano_input_variables()

        parameter_shapes = model.get_weight_shapes()
        self.initialize_running_average(parameter_shapes)        
        
        loss,_ = theano.scan(model.theano_loss,
                             sequences=input_variable_list)

        sumloss = T.sum(loss)
        regularization = model.theano_l2_regularization()
        sumloss += self.regularization_parameter * regularization
        
        gradient = T.grad(sumloss, wrt=model.get_weights())

        update_list = [None]*(len(gradient)*2)
        for i in range(len(gradient)):
            squared_gradient = gradient[i] * gradient[i]
            new_running_average = self.decay_rate * self.running_average[i] + (1 - self.decay_rate) * squared_gradient
            rmsgrad = T.sqrt(self.running_average[i] + self.epsillon)
            
            update_list[i] = (model.get_weights()[i], self.update_parameter(model.get_weights()[i], gradient[i], rmsgrad))            
            update_list[i+len(gradient)] = (self.running_average[i], new_running_average)

        return theano.function(inputs=input_variable_list, outputs=[sumloss, regularization], updates=update_list)



