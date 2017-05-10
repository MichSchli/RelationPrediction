import tensorflow as tf

"""
Class representing an hierarchically organized model to be initialized in a dependency injection-like manner.
"""


class Model:
    score_all_subjects_graph = None
    score_all_objects_graph = None
    score_graph = None
    session = None
    next_component=None
    save_iter=0
    saver=None

    def __init__(self, next_component, settings):
        self.next_component = next_component
        self.settings = settings

        self.entity_count = int(self.settings['EntityCount'])
        self.relation_count = int(self.settings['RelationCount'])
        self.edge_count = int(self.settings['EdgeCount'])

        self.parse_settings()

    def parse_settings(self):
        pass

    def save(self, save_path):
        variables_to_save = self.get_weights()

        if self.saver is None:
            self.saver = tf.train.Saver(var_list=variables_to_save)

        print("saving...")
        self.saver.save(self.session, save_path, global_step=self.save_iter)
        self.saver.restore(self.session, save_path+"-"+str(self.save_iter))
        self.save_iter += 1


    '''
    High-level functions:
    '''

    def score(self, triplets):
        if self.score_graph is None:
            self.score_graph = self.predict()

        if self.needs_graph():
            d = {self.get_test_input_variables()[0]: self.train_triplets,
                 self.get_test_input_variables()[1]: triplets}
        else:
            d = {self.get_test_input_variables()[0]: triplets}

        return self.session.run(self.score_graph, feed_dict=d)


    def score_all_subjects(self, triplets):
        if self.score_all_subjects_graph is None:
            self.score_all_subjects_graph = self.predict_all_subject_scores()

        if self.needs_graph():
            d = {self.get_test_input_variables()[0]: self.test_graph,
                 self.get_test_input_variables()[1]: triplets}
        else:
            d = {self.get_test_input_variables()[0]: triplets}

        return self.session.run(self.score_all_subjects_graph, feed_dict=d)

    def score_all_objects(self, triplets):
        if self.score_all_objects_graph is None:
            self.score_all_objects_graph = self.predict_all_object_scores()

        if self.needs_graph():
            d = {self.get_test_input_variables()[0]: self.test_graph,
                 self.get_test_input_variables()[1]: triplets}
        else:
            d = {self.get_test_input_variables()[0]: triplets}

        return self.session.run(self.score_all_objects_graph, feed_dict=d)

    '''
    '''

    def register_for_test(self, triplets):
        self.test_graph = triplets

    def preprocess(self, triplets):
        self.train_triplets = triplets
        pass #return self.__local_run_delegate__('preprocess', triplets)

    def initialize_train(self):
        return self.__local_run_delegate__('initialize_train')

    def get_weights(self):
        return self.__local_expand_delegate__('get_weights')

    def set_variable(self, name, value):
        return self.__local_run_delegate__('set_variable', name, value)

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

    def predict(self):
        return self.__delegate__('predict')

    def predict_all_subject_scores(self):
        return self.__delegate__('predict_all_subject_scores')

    def predict_all_object_scores(self):
        return self.__delegate__('predict_all_object_scores')

    def get_graph(self):
        return self.__delegate__('get_graph')

    def get_additional_ops(self):
        return self.__local_expand_delegate__('get_additional_ops')

    def needs_graph(self):
        if self.next_component is None:
            return False
        else:
            return self.next_component.needs_graph()

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
    def __local_expand_delegate__(self, name, *args, base=None):
        if base is None:
            base = []
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




