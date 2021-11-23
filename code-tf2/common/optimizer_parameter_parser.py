
class Parser():

    sample_transform_function = None
    save_function = None
    early_stopping_score_function = None
    additional_ops = []
    
    def __init__(self, optimizer_settings):
        self.settings = optimizer_settings

    def minibatches(self):
        if 'BatchSize' in self.settings:
            return ('Minibatches',{
                'batch_size':int(self.settings['BatchSize']),
                'contiguous_sampling':False
            })
        else:
            return None

    def set_sample_transform_function(self, function):
        self.sample_transform_function = function

    def set_save_function(self, function):
        self.save_function = function

    def set_early_stopping_score_function(self, function):
        self.early_stopping_score_function = function

    def set_additional_ops(self, ops):
        self.additional_ops = ops

    def sample_transform(self):
        if self.sample_transform_function is not None:
            return ('SampleTransformer', {'transform_function': self.sample_transform_function})
        else:
            return None

    def gradient_clipping(self):
        if 'MaxGradientNorm' in self.settings:
            return ('GradientClipping', {
                'max_norm': float(self.settings['MaxGradientNorm'])
            })
        else:
            return None

    def iteration_counter(self):
        if 'MaxIterations' in self.settings:
            return ('IterationCounter', {
                'max_iterations': int(self.settings['MaxIterations'])
            })
        else:
            return None

    def optimization_algorithm(self):
        algorithm_settings = self.settings['Algorithm']
        name = algorithm_settings['Name']

        d = {}
        for setting in algorithm_settings:
            if setting != 'Name':
                d[setting] = float(algorithm_settings[setting])

        return (name, d)

    def train_loss_reporter(self):
        if 'ReportTrainLossEvery' in self.settings:
            return ('TrainLossReporter', {
                'evaluate_every_n': int(self.settings['ReportTrainLossEvery'])
            })
        else:
            return None

    def early_stopping(self):
        if 'EarlyStopping' in self.settings:
            early_stopping_settings = self.settings['EarlyStopping']
        else:
            return None

        if 'BurninPhaseDuration' in early_stopping_settings:
            b = int(early_stopping_settings['BurninPhaseDuration'])
        else:
            b = 0

        return ('EarlyStopper', {
            'criteria':'score_validation_data',
            'evaluate_every_n': int(early_stopping_settings['CheckEvery']),
            'scoring_function': self.early_stopping_score_function,
            'comparator':lambda current, prev: current > prev,
            'burnin': b})

    def model_saving(self):
        n=1

        if 'SaveEveryN' in self.settings:
            n = int(self.settings['SaveEveryN'])
        elif 'EarlyStopping' in self.settings:
            n = int(self.settings['EarlyStopping']['CheckEvery'])
        
        return ('ModelSaver', {
            'save_function': self.save_function,
            'model_path': self.settings['ExperimentName'],
            'save_every_n': n})


    def get_additional_ops(self):
        return [('AdditionalOp', {'op': op}) for op in self.additional_ops]
    
    def get_parametrization(self):
        params = [self.minibatches(),
                  self.sample_transform(),
                  self.iteration_counter(),
                  self.gradient_clipping()
        ]

        params += self.get_additional_ops()

        params += [self.optimization_algorithm(),
                  self.train_loss_reporter(),
                  self.early_stopping(),
                  self.model_saving()
        ]
        
        return [p for p in params if p is not None]
