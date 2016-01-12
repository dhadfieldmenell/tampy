from IPython import embed as shell

class Parameter:
    """
    Parameters fall into one of two categories: objects or symbols. Objects are things that
    exist in the environment with some pose, and symbols are symbolic references.
    """
    def __init__(self, *args):
        raise NotImplementedError("Must instantiate either Object or Symbol.")

    def get_type(self):
        return self.__class__.__name__

    def is_symbol(self):
        return False

class Object(Parameter):
    """
    Objects have a name and a pose in the world at each timestep (pose is a d-by-T table, which
    we refer to as its trajectory table).
    """
    def __init__(self, name):
        self.name = name
        self.pose = "undefined"

    def is_defined(self):
        return self.pose != "undefined"

class Symbol(Parameter):
    """
    Symbols have a name and a value at each timestep (value is a d-by-T table, which
    we refer to as its trajectory table).
    """
    def __init__(self, name):
        self.name = name
        self.value = "undefined"

    def is_defined(self):
        return self.value != "undefined"

    def is_symbol(self):
        return True

class Can(Object):
    pass

class Target(Object):
    pass

class Robot(Object):
    pass

class Manip(Object):
    pass
