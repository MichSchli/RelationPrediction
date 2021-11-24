from optimization.abstract import IOptimizer
import numpy as np
import tensorflow as tf

class GradientDescent(IOptimizer):

    learning_rate = None

    def valid(self):
        return self.learning_rate is not None

    def process_update_function(self, gradient_function, parameters_to_optimize):
        opt_func = tf.compat.v1.train.GradientDescentOptimizer(self.learning_rate)
        optimizer = opt_func.apply_gradients(zip(gradient_function, parameters_to_optimize))

        return optimizer

class AdditionalOp(IOptimizer):
    op = None

    def valid(self):
        return self.op is not None

    def get_additional_ops(self):
        return self.next_component.get_additional_ops() + [self.op]

class Adam(IOptimizer):

    learning_rate = None
    historical_moment_weight = 0.9
    historical_second_moment_weight = 0.999

    def valid(self):
        return self.learning_rate is not None

    def process_update_function(self, gradient_function, parameters_to_optimize):
        opt_func = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate,
                                          beta1=self.historical_moment_weight,
                                          beta2=self.historical_second_moment_weight)
        optimizer = opt_func.apply_gradients(zip(gradient_function, parameters_to_optimize))

        return optimizer

class AdaGrad(IOptimizer):

    learning_rate = None
    
    def valid(self):
        return self.learning_rate is not None

    def process_update_function(self, gradient_function, parameters_to_optimize):
        opt_func = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.learning_rate)
        optimizer = opt_func.apply_gradients(zip(gradient_function, parameters_to_optimize))

        return optimizer

    
class GradientClipping(IOptimizer):

    max_norm = None

    def valid(self):
        return self.max_norm is not None

    def process_gradient_function(self, loss_function, parameters_to_optimize):
        gradient = self.next_component.process_gradient_function(loss_function, parameters_to_optimize)
        clipped,_ = tf.clip_by_global_norm(gradient, self.max_norm)
        return clipped


class ModelSaver(IOptimizer):

    model_path = None
    save_function = None
    save_every_n = 1

    def valid(self):
        return self.model_path is not None and self.save_function is not None

    def postprocess(self, loss):
        value_of_next = self.next_component.postprocess(loss)

        if value_of_next == 'stop':
            return 'stop'
        
        if self.iteration % self.save_every_n == 0:
            self.save_function(self.model_path)

        return value_of_next
                

