class Parameter:
    """
    Parameters fall into one of two categories: objects or symbols. Objects are things that
    exist in the environment with some pose, and symbols are symbolic references.
    """
    def __init__(self, *args):
        raise NotImplementedError("Must instantiate either Object or Symbol.")

class Object(Parameter):
    """
    Objects have a name and a pose in the world at each timestep (pose is a d-by-T table, which
    we refer to as its trajectory table).
    """
    def __init__(self, name, pose):
        self.name = name
        self.pose = pose

class Symbol(Parameter):
    """
    Symbols have a name and a value at each timestep (value is a d-by-T table, which
    we refer to as its trajectory table).
    """
    def __init__(self, name, value):
        self.name = name
        self.value = value
