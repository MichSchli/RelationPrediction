
"""
Class representing an hierarchically organized model to be initialized in a dependency injection-like manner.
"""


class Model:
    score_all_subjects_graph = None
    score_all_objects_graph = None
    session = None

    def __init__(self, next_component, settings):
        self.next_component = next_component
        self.settings = settings

        self.entity_count = int(self.settings['EntityCount'])
        self.relation_count = int(self.settings['RelationCount'])

        self.parse_settings()

    def parse_settings(self):
        pass

    '''
    High-level functions:
    '''
    def score_all_subjects(self, triplets):
        if self.score_all_subjects_graph is None:
            self.score_all_subjects_graph = self.predict_all_subject_scores()

        return self.session.run(self.score_all_subjects_graph, feed_dict={self.get_test_input_variables()[0]: triplets})

    def score_all_objects(self, triplets):
        if self.score_all_objects_graph is None:
            self.score_all_objects_graph = self.predict_all_object_scores()

        return self.session.run(self.score_all_objects_graph, feed_dict={self.get_test_input_variables()[0]: triplets})

    '''
    '''

    def preprocess(self, triplets):
        pass #return self.__local_run_delegate__('preprocess', triplets)

    def initialize_train(self):
        return self.__local_run_delegate__('initialize_train')

    def get_weights(self):
        return self.__local_expand_delegate__('get_weights')

    def get_train_input_variables(self):
        return self.__local_expand_delegate__('get_train_input_variables')

    def get_test_input_variables(self):
        return self.__local_expand_delegate__('get_test_input_variables')

    def get_loss(self, mode='train'):
        return self.__delegate__('get_loss', mode)

    def get_regularization(self):
        return self.__local_expand_delegate__('get_regularization', base=0)

    def get_all_subject_codes(self, mode='train'):
        return self.__delegate__('get_all_subject_codes', mode)

    def get_all_object_codes(self, mode='train'):
        return self.__delegate__('get_all_object_codes', mode)

    def get_all_codes(self, mode='train'):
        return self.__delegate__('get_all_codes', mode)

    def predict_all_subject_scores(self):
        return self.__delegate__('predict_all_subject_scores')

    def predict_all_object_scores(self):
        return self.__delegate__('predict_all_object_scores')

    '''
    Delegate function to the highest-level component with a definition:
    '''
    def __delegate__(self, name, *args):
        if self.next_component is not None:
            function = getattr(self.next_component, name)
            return function(*args)
        return None

    '''
    Run the function locally if it exists, then delegate to the next component:
    '''
    def __local_run_delegate__(self, name, *args):
        local_function_name = 'local_' + name
        if hasattr(self, local_function_name):
            local_function = getattr(self, local_function_name)
            local_function(*args)

        if self.next_component is not None:
            function = getattr(self.next_component, name)
            function(*args)

    '''
    Run the function locally if it exists, then compose with the next component through addition:
    '''
    def __local_expand_delegate__(self, name, *args, base=[]):
        local_function_name = 'local_'+name
        if hasattr(self, local_function_name):
            local_function = getattr(self, local_function_name)
            local_result = local_function(*args)
        else:
            local_result = base

        if self.next_component is not None:
            function = getattr(self.next_component, name)
            return function(*args) + local_result
        return local_result




