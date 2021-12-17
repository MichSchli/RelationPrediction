from optimization.abstract import IOptimizer
import random
import numpy as np

class IterationCounter(IOptimizer):

    max_iterations = None
    iterations = 0

    def valid(self):
        return self.max_iterations is not None

    def next_batch(self):
        if self.iterations < self.max_iterations:
            self.iterations += 1
            return self.next_component.next_batch()
        else:
            return None

class Minibatches(IOptimizer):

    batch_size = None
    contiguous_sampling = None

    current_batch = None

    def valid(self):
        return self.batch_size is not None and self.contiguous_sampling is not None

    def next_batch(self):
        if self.contiguous_sampling:
            return self.__contiguous_sample()
        else:
            return self.__random_sample()
    
    def __contiguous_sample(self):
        if current_batch is None:
            current_batch = self.next_component.next_batch()
        pass

    def __random_sample(self):
        data = self.next_component.next_batch()
        n_total = len(data)

        sample = random.sample(range(n_total), self.batch_size)

        return [data[i] for i in sample]


class SampleTransformer(IOptimizer):

    transform_function = None

    def valid(self):
        return self.transform_function is not None

    def process_data(self, training_data):
        data = self.next_component.process_data(training_data)
        return self.transform_function(data)

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

    
class TrainLossReporter(IOptimizer):
    evaluate_every_n = 1

    cummulative_loss = 0

    def valid(self):
        return True

    def postprocess(self, loss):
        value_of_next = self.next_component.postprocess(loss)

        if value_of_next == 'stop':
            return 'stop'
        
        self.cummulative_loss += loss

        if self.iteration == 1:
            self.cummulative_loss = 0
            print("Initial loss: "+str(loss))
            return value_of_next
                  
        if self.iteration % self.evaluate_every_n == 1:
            average_loss = self.cummulative_loss / float(self.evaluate_every_n)
            self.cummulative_loss = 0

            begin_iteration = self.iteration - self.evaluate_every_n
            end_iteration = self.iteration - 1
            print("Average train loss for iteration "
                  + str(begin_iteration)
                  + "-"
                  + str(end_iteration)
                  + ": "
                  + str(average_loss))

        return value_of_next

            
class EarlyStopper(IOptimizer):

    criteria = None
    evaluate_every_n = 1
    
    previous_validation_score = None
    burnin = 0
    
    def valid(self):
        if self.criteria is None:
            return False

        if self.criteria == 'score_validation_data' and self.scoring_function is None:
            return False

        if self.criteria == 'score_validation_data' and self.comparator is None:
            return False
        
        return self.evaluate_every_n is not None

    def postprocess(self, loss):
        value_of_next = self.next_component.postprocess(loss)

        if value_of_next == 'stop':
            return 'stop'
        
        if self.iteration % self.evaluate_every_n == 0:
            if self.criteria == 'score_validation_data':
                validation_score = self.scoring_function(self.validation_data)

                print("Tested validation score at iteration "+str(self.iteration)+". Result: "+str(validation_score))
                if self.previous_validation_score is not None:
                    if not self.comparator(validation_score, self.previous_validation_score):
                        if self.iteration > self.burnin:
                            print("Stopping criterion reached.")
                        
                            return 'stop'
                        else:
                            print("Ignoring criterion while in burn-in phase.")

                self.previous_validation_score = validation_score

        return value_of_next
