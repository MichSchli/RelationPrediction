class Settings():

    
    def __init__(self):
        self.__dict__ = {}

    def parse(self, filename):
        f = open(filename, 'r')
        self.internal_parse(list(f))

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)

    def __contains__(self, key):
        return key in self.__dict__

    def __iter__(self):
        return self.__dict__.__iter__()

    def merge(self, other_settings):
        self.__dict__.update(other_settings.__dict__)

    def put(self, key, value):
        self.__dict__[key] = value
    
    def internal_parse(self, lines, indent=0):
        for i,line in enumerate(lines):
            if line.strip():
                indent_level = self.__count_indents__(line)

                if indent_level < indent:
                    break

                if indent_level > indent:
                    continue

                line = line.strip()

                if line.startswith('['):
                    line = line[1:-1]
                    self.__dict__[line] = Settings()
                    self.__dict__[line].internal_parse(lines[i+1:], indent=indent+1)
                else:
                    parts = [p.strip() for p in line.split('=')]
                    self.__dict__[parts[0]] = parts[1]

                


    def __count_indents__(self, line):
        for i,c in enumerate(line):
            if c != '\t':
                return i
    
    def __getitem__(self, key): 
        return self.__dict__[key]

def read(filename):
    settings = Settings()
    settings.parse(filename)
    return settings
