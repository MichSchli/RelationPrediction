import optimization.tensorflow_backend.algorithms as tensorflow_algorithms
import optimization.theano_backend.algorithms as theano_algorithms
import optimization.shared.algorithms as shared_algorithms
import theano
from theano import tensor as T
from optimization.abstract import BaseOptimizer
import numpy as np
import tensorflow as tf

'''
Partial class for shared architecture between theano/tensorflow optimizers.
'''

class Optimizer():

    def __init__(self, stack):
        self.stack = stack

    def fit(self, training_data, validation_data=None):
        self.stack.set_training_data(training_data)
        if validation_data is not None:
            self.stack.set_validation_data(validation_data)

        self.initialize_for_fitting()

        i = 0
        next_batch = self.stack.next_batch()
        while(next_batch is not None):
            i+=1
            self.stack.set_iteration(i)

            processed_batch = self.stack.process_data(next_batch)
            train_loss = self.update_from_batch(processed_batch)
            
            if self.stack.postprocess(train_loss) == 'stop':
                print("Stopping training.")
                break

            next_batch = self.stack.next_batch()
        
class TensorflowOptimizer(Optimizer):

    def set_placeholders(self, placeholders):
        self.placeholders = placeholders

    def set_session(self, session):
        self.session = session
        
    def compute_functions(self, loss_function, parameters_to_optimize):
        self.loss_function = self.stack.process_loss_function(loss_function)
        self.gradient_function = self.stack.process_gradient_function(self.loss_function, parameters_to_optimize)
        self.update_function = self.stack.process_update_function(self.gradient_function, parameters_to_optimize)
        self.variables = parameters_to_optimize
        
    def loss(self, placeholder_input):
        #self.session = tf.Session()
        placeholder_input = self.stack.process_data(placeholder_input)
        init_op = tf.compat.v1.initialize_all_variables()
        self.session.run(init_op)
        
        feed_dict = dict(zip(self.placeholders, placeholder_input))
        return self.session.run(self.loss_function, feed_dict=feed_dict) 
    
    def gradients(self, placeholder_input):
        #self.session = tf.Session()

        placeholder_input = self.stack.process_data(placeholder_input)
        init_op = tf.compat.v1.initialize_all_variables()
        self.session.run(init_op)
        
        feed_dict = dict(zip(self.placeholders, placeholder_input))
        return self.session.run(self.gradient_function, feed_dict=feed_dict)

    def initialize_for_fitting(self):
        #self.session = tf.Session()
        self.stack.set_session(self.session)
        
        init_op = tf.compat.v1.initialize_all_variables()
        self.session.run(init_op)

    def update_from_batch(self, processed_batch):        
        feed_dict = dict(zip(self.placeholders, processed_batch))

        adds = self.stack.get_additional_ops()
        upd = self.session.run([self.update_function, self.loss_function, adds],
                                       feed_dict=feed_dict)

        return upd[1]

class TheanoOptimizer(Optimizer):

    __loss__ = None
    __update__ = None
    __parameters__ = None
    
    def __init__(self, stack):
        self.stack = stack

    def set_loss_function(self, loss_function):
        self.__loss__ = self.stack.process_loss_function(loss_function)
        
    def set_parameters_to_optimize(self, parameters_to_optimize):
        self.__parameters__ = parameters_to_optimize
        
    def compute_update_function(self, input_params):
        raw_update = self.stack.theano_process_update_function(self.__parameters__, self.__loss__)
        self.__update__ = theano.function(inputs=input_params, outputs=self.__loss__, updates=raw_update)
        
        self.__loss__ = theano.function(inputs=input_params, outputs=self.__loss__)

    def initialize_for_fitting(self):
        pass
    
    def update_from_batch(self, processed_batch):
        return self.__update__(*tuple(processed_batch))

    '''
    def fit(self, training_data, validation_data=None):
        self.stack.set_training_data(training_data)
                  
        i = 0
        train_loss = 0
        next_batch = self.stack.next_batch()
        while(next_batch is not None):
            i+=1
            
            message = self.stack.get_message()
            if message is not None:
                print(message)

            processed_batch = self.stack.process_data(next_batch)
            train_loss += self.__update__(*tuple(processed_batch))
            self.stack.postprocess()

            
            if i % 100 == 0:
                print(i)
                print(train_loss/100)
                train_loss = 0

            if i == 1:
                print(train_loss)
                
            message = self.stack.get_message()
            if message is not None:
                print(message)
            
            next_batch = self.stack.next_batch()

    '''
    
def __from_component(component_name, backend='theano'):
    print(component_name)
    if component_name == "GradientDescent":
        if backend == 'theano':
            return theano_algorithms.GradientDescent
        elif backend == 'tensorflow':
            return tensorflow_algorithms.GradientDescent
    
    if component_name == "Minibatches":
        return shared_algorithms.Minibatches

    if component_name == "IterationCounter":
        return shared_algorithms.IterationCounter

    if component_name == "SampleTransformer":
        return shared_algorithms.SampleTransformer

    if component_name == "GradientClipping":
        if backend == 'theano':
            return theano_algorithms.GradientClipping
        elif backend == 'tensorflow':
            return tensorflow_algorithms.GradientClipping

    if component_name == "EarlyStopper":
        return shared_algorithms.EarlyStopper
        
    if component_name == "AdaGrad":
        if backend == 'theano':
            return theano_algorithms.AdaGrad
        elif backend == 'tensorflow':
            return tensorflow_algorithms.AdaGrad

    if component_name == "RmsProp":
        if backend == 'theano':
            return theano_algorithms.RmsProp
        elif backend == 'tensorflow':
            return tensorflow_algorithms.RmsProp

    if component_name == "Adam":
        if backend == 'theano':
            return theano_algorithms.Adam
        elif backend == 'tensorflow':
            return tensorflow_algorithms.Adam

    if component_name == "ModelSaver":
        return shared_algorithms.ModelSaver

    if component_name == "TrainLossReporter":
        return shared_algorithms.TrainLossReporter

    if component_name == "AdditionalOp":
        return tensorflow_algorithms.AdditionalOp
        
    
def __construct_optimizer(settings, backend='theano'):
    optimizer = BaseOptimizer()
    for component, parameters in settings:
        optimizer = __from_component(component, backend=backend)(optimizer, parameters)

    #TODO: Better error handling
    if not optimizer.verify():
        print("Construction failed.")

    if backend == 'theano':
        return TheanoOptimizer(optimizer)
    elif backend == 'tensorflow':
        return TensorflowOptimizer(optimizer)

def build_theano(loss_function, parameters_to_optimize, settings, input_params):
    optimizer = __construct_optimizer(settings, backend='theano')
    
    optimizer.set_loss_function(loss_function)
    optimizer.set_parameters_to_optimize(parameters_to_optimize)
    optimizer.compute_update_function(input_params)
    
    return optimizer

def build_tensorflow(loss_function, parameters_to_optimize, settings, placeholders):
    optimizer = __construct_optimizer(settings, backend='tensorflow')
    
    optimizer.compute_functions(loss_function, parameters_to_optimize)
    optimizer.set_placeholders(placeholders)
    
    return optimizer


if __name__ == '__main__':
    X = tf.compat.v1.placeholder(tf.float32, shape=(None,2))
    Y = tf.compat.v1.placeholder(tf.float32, shape=(None))

    n_hidden = 10
    
    W1 = tf.Variable(np.random.randn(2,n_hidden).astype(np.float32))
    W2 = tf.Variable(np.random.randn(n_hidden,1).astype(np.float32))

    b1 = tf.Variable(np.random.randn(n_hidden).astype(np.float32))
    b2 = tf.Variable(np.random.randn(1).astype(np.float32))
    
    hidden = tf.tanh(tf.matmul(X, W1)+b1)
    energy = tf.matmul(hidden, W2)+b2
    output = tf.sigmoid(energy)

    #loss = tf.reduce_mean(-Y * tf.log(output+1e-10) - (1- Y) * tf.log(1-output+1e-10))
    loss = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(energy, Y))
    
    parameters = [#('Minibatches', {'batch_size':2, 'contiguous_sampling':False}),
                  ('IterationCounter', {'max_iterations':5000}),
                  ('GradientClipping', {'max_norm':1}),
                  #('Adam', {'learning_rate':0.000001, 'historical_moment_weight':0.9, 'historical_gradient_weight':0.999}),                  
                  ('GradientDescent', {'learning_rate':0.01})
    ]
    
    opt = tfbuild(loss, parameters, [X,Y])
    opt.predict = output

    xor_toy_problem = [[1,0],[0,0],[0,1],[1,1]]

    xor_toy_labels = [[1],[0],[1],[0]]
    
    print(opt.loss([xor_toy_problem, xor_toy_labels]))
    opt.fit([xor_toy_problem, xor_toy_labels])
    print(opt.loss([xor_toy_problem, xor_toy_labels]))
    
