import theano
from theano import tensor as T
import gc
import numpy as np
import random
from Converge import optimize as converge

class AbstractBatchOptimizer():

    max_iterations = 100000
    validate_after_iterations = 5000
    batch_size = 4831

    def __init__(self):
        pass

    def converge_train(self, model, train_triplets, valid_triplets, model_path):
        transform = lambda x: model.process_train_triplets(x, train_triplets+valid_triplets)
        
        parameters = [('Minibatches', {'batch_size':5, 'contiguous_sampling':False}),
                      ('SampleTransformer', {'transform_function': transform}),
                      ('IterationCounter', {'max_iterations':5000}),
                      ('GradientClipping', {'max_norm':1}),
                      ('Adam', {'learning_rate':0.05, 'historical_moment_weight':0.9, 'historical_gradient_weight':0.999}),
                      ('ModelSaver', {'save_function': model.save, 'model_path': model_path})]

        weights = model.get_weights()
        input_variable_list = model.get_theano_input_variables()
        loss = model.theano_batch_loss(input_variable_list)

        opt = converge.build(loss,weights,input_variable_list, parameters)
        
        print(opt.loss(valid_triplets[:200]))
        opt.fit(train_triplets)
        print(opt.loss(valid_triplets[:200]))
    
    
    def train(self, model, train_triplets, valid_triplets, model_path):

        self.converge_train(model, train_triplets, valid_triplets, model_path)
        exit()
        update_function = self.compute_update_function(model)
        loss_function = model.compute_batch_loss_function()

        validation_samples = model.process_train_triplets(valid_triplets, valid_triplets + train_triplets, disable_saving=True)
        previous_validation_loss = loss_function(*validation_samples)
        train_loss = 0
        report = 100
        
        for i in range(self.max_iterations):
            if i % report == 0:
                print("Running optimizer at iteration "+str(i + 1)+"...", end="\n", flush=True)
                
            if i % self.validate_after_iterations == 0:
                print("Current validation loss: "+str(previous_validation_loss))
                
            processed_samples = model.process_train_triplets(random.sample(train_triplets, self.batch_size), train_triplets)
            train_loss += update_function(*processed_samples)

            if i % report == 0:
                print("Train loss: "+str(train_loss/report))
                train_loss = 0

            if i % self.validate_after_iterations == 0:
                validation_loss = loss_function(*validation_samples)
                
                #if validation_loss >= previous_validation_loss:
                #    print("Stopping early. Validation loss: "+str(validation_loss))
                #    break
                #else:
                previous_validation_loss = validation_loss

            model.save(model_path)

                
    def chunk_iterate(self, length):
        for i in range(0, length, self.batch_size):
            yield i, i+self.batch_size
    
            
class BatchGradientDescent(AbstractBatchOptimizer):

    learning_rate = 0.1
    regularization_parameter = 0.01
    
    def __init__(self):
        pass

    def update_parameter(self, parameter, gradient):
        return parameter - (self.learning_rate / self.batch_size) * gradient
    
    def compute_update_function(self, model):
        input_variable_list = model.get_theano_input_variables()
        
        loss = model.theano_batch_loss(input_variable_list)
        
        regularization = model.theano_l2_regularization()
        loss += self.regularization_parameter * regularization
        
        gradient = T.grad(loss, wrt=model.get_weights())

        update_list = [None]*len(gradient)
        for i in range(len(gradient)):
            update_list[i] = (model.get_weights()[i], self.update_parameter(model.get_weights()[i], gradient[i]))
        
        return theano.function(inputs=input_variable_list, outputs=[loss, regularization], updates=update_list)


class Adam(AbstractBatchOptimizer):

    learning_rate = 0.3

    epsillon = 10**(-8)
    beta_1 = 0.9
    beta_2 = 0.999
    
    def __init__(self):
        pass

    def update_parameter(self, parameter, m, v):
        return parameter - (self.learning_rate) * m / (T.sqrt(v) + self.epsillon)

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
        
        loss = model.theano_batch_loss(input_variable_list)

        gradient = T.grad(loss, wrt=model.get_weights())

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
        return theano.function(inputs=input_variable_list, outputs=loss, updates=update_list)

    
class RmsProp(AbstractBatchOptimizer):

    learning_rate = 0.5
    regularization_parameter = 0.01
    epsillon = 10**(-8)
    decay_rate = 0.9
    max_gradient_norm = 1.0
    
    def __init__(self):
        pass

    def update_parameter(self, parameter, gradient, rmsgrad):
        return parameter - (self.learning_rate / rmsgrad) * gradient

    def initialize_running_average(self, parameter_shape_list):
        self.running_average = [None]*len(parameter_shape_list)
        
        for i,parameter_shape in enumerate(parameter_shape_list):
            self.running_average[i] = theano.shared(np.zeros(parameter_shape).astype(np.float32))
            
    
    def compute_update_function(self, model):
        input_variable_list = model.get_theano_input_variables()

        parameter_shapes = model.get_weight_shapes()
        self.initialize_running_average(parameter_shapes)        
        
        loss = model.theano_batch_loss(input_variable_list)
        
        gradient = list(T.grad(loss, wrt=model.get_weights()))
        for i,grad in enumerate(gradient):
            norm = T.sqrt((grad * grad).sum())
            gradient[i] = grad * T.minimum(1, self.max_gradient_norm / norm)
            
        update_list = [None]*(len(gradient)*2)
        for i in range(len(gradient)):
            squared_gradient = gradient[i] * gradient[i]
            new_running_average = self.decay_rate * self.running_average[i] + (1 - self.decay_rate) * squared_gradient
            rmsgrad = T.sqrt(new_running_average + self.epsillon)
            
            update_list[i] = (model.get_weights()[i], self.update_parameter(model.get_weights()[i], gradient[i], rmsgrad))            
            update_list[i+len(gradient)] = (self.running_average[i], new_running_average)

        return theano.function(inputs=input_variable_list, outputs=loss, updates=update_list)


class AdaGrad(AbstractBatchOptimizer):

    learning_rate = 0.5
    epsillon = 10**(-8)
    max_gradient_norm = 1.0
    
    def __init__(self):
        pass

    def update_parameter(self, parameter, gradient, scaling):
        return parameter - (self.learning_rate / scaling) * gradient

    def initialize_running_average(self, parameter_shape_list):
        self.historical_gradient = [None]*len(parameter_shape_list)
        
        for i,parameter_shape in enumerate(parameter_shape_list):
            self.historical_gradient[i] = theano.shared(np.zeros(parameter_shape).astype(np.float32))
            
    
    def compute_update_function(self, model):
        input_variable_list = model.get_theano_input_variables()

        parameter_shapes = model.get_weight_shapes()
        self.initialize_running_average(parameter_shapes)        
        
        loss = model.theano_batch_loss(input_variable_list)
        
        gradient = list(T.grad(loss, wrt=model.get_weights()))
        for i,grad in enumerate(gradient):
            norm = T.sqrt((grad * grad).sum())
            gradient[i] = grad * T.minimum(1, self.max_gradient_norm / norm)
        
        update_list = [None]*(len(gradient)*2)
        for i in range(len(gradient)):
            new_historical_gradient = self.historical_gradient[i] + gradient[i] * gradient[i]
            scaling = T.sqrt(new_historical_gradient + self.epsillon)
            
            update_list[i] = (model.get_weights()[i], self.update_parameter(model.get_weights()[i], gradient[i], scaling))            
            update_list[i+len(gradient)] = (self.historical_gradient[i], new_historical_gradient)

        return theano.function(inputs=input_variable_list, outputs=loss, updates=update_list)

    
class AdaDelta(AbstractBatchOptimizer):

    learning_rate = 1.0
    epsillon = 10**(-8)
    decay_rate = 0.9
    max_gradient_norm = 1.0
    
    def __init__(self):
        pass
    
    def initialize_running_average(self, parameter_shape_list):
        self.historical_gradient = [None]*len(parameter_shape_list)
        self.historical_updates = [None]*len(parameter_shape_list)
        
        for i,parameter_shape in enumerate(parameter_shape_list):
            self.historical_gradient[i] = theano.shared(np.zeros(parameter_shape).astype(np.float32))
            self.historical_updates[i] = theano.shared(np.zeros(parameter_shape).astype(np.float32))

            
    def compute_update_function(self, model):
        input_variable_list = model.get_theano_input_variables()

        parameter_shapes = model.get_weight_shapes()
        self.initialize_running_average(parameter_shapes)        
        
        loss = model.theano_batch_loss(input_variable_list)

        gradient = list(T.grad(loss, wrt=model.get_weights()))
        for i,grad in enumerate(gradient):
            norm = T.sqrt((grad * grad).sum())
            gradient[i] = grad * T.minimum(1, self.max_gradient_norm / norm)
        
        update_list = [None]*(len(gradient)*3)
        for i in range(len(gradient)):
            new_historical_gradient = self.decay_rate * self.historical_gradient[i] + (1 - self.decay_rate) * gradient[i] * gradient[i]
            rms_gradient = T.sqrt(new_historical_gradient + self.epsillon)
            rms_updates = T.sqrt(self.historical_updates[i] + self.epsillon)

            update = - self.learning_rate * rms_updates / rms_gradient * gradient[i]
            new_historical_update = self.decay_rate * self.historical_updates[i] + (1 - self.decay_rate) * (update * update)

            
            update_list[i] = (model.get_weights()[i], model.get_weights()[i] + update)            
            update_list[i+len(gradient)] = (self.historical_gradient[i], new_historical_gradient)
            update_list[i+len(gradient)*2] = (self.historical_updates[i], new_historical_update)


        return theano.function(inputs=input_variable_list, outputs=loss, updates=update_list)
