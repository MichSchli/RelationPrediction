from model import Model
"""
Class representing an hierarchically organized model to be initialized in a dependency injection-like manner.
"""


class SplitModel(Model):

    def __init__(self, next_component_list, settings):
        self.next_components = next_component_list
        self.settings = settings

        self.entity_count = int(self.settings['EntityCount'])
        self.relation_count = int(self.settings['RelationCount'])

        self.parse_settings()

    def needs_graph(self):
        for component in self.next_components:
            if component.needs_graph():
                return True
        return False

    '''
    Delegate function to the highest-level component with a definition:
    '''
    def __delegate__(self, name, *args):
        raise Exception

    '''
    Run the function locally if it exists, then delegate to the next component:
    '''
    def __local_run_delegate__(self, name, *args):
        local_function_name = 'local_' + name
        if hasattr(self, local_function_name):
            local_function = getattr(self, local_function_name)
            local_function(*args)

        for component in self.next_components:
            function = getattr(component, name)
            function(*args)

    '''
    Run the function locally if it exists, then compose with the next component through addition:
    '''
    def __local_expand_delegate__(self, name, *args, base=None):
        should_set_add = False

        if base is None:
            #total hack
            should_set_add = True

            base = []

        local_function_name = 'local_'+name
        if hasattr(self, local_function_name):
            local_function = getattr(self, local_function_name)
            local_result = local_function(*args)
        else:
            local_result = base

        for component in self.next_components:
            function = getattr(component, name)
            local_result += function(*args)

        # TODO: Total hack. Don't publish this :D
        if should_set_add:
            local_result = list(set(local_result))

        return local_result




