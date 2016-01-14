from IPython import embed as shell

class Parameter(object):
    """
    Parameters fall into one of three categories: objects, symbols, or the workspace. Objects are things that
    exist in the environment with some pose, and symbols are symbolic references. The workspace parameter
    holds information about the environment that's useful for planning.
    """
    def __init__(self, *args):
        raise NotImplementedError("Must instantiate either Object or Symbol.")

    def get_type(self):
        return self.__class__.__name__

    def is_symbol(self):
        return False

    def __repr__(self):
        return "%s - %s"%(self.name, self.get_type())

class Object(Parameter):
    """
    Objects must have at minimum a name and a pose attribute (a d-by-T table, which we refer to as the
    trajectory table), which is set to "undefined" if the object's pose is
    not defined. The attributes for the objects are defined in the configuration files.
    attrs is a dictionary from instance attribute name to the argument to pass into the __init__ method
    for the class stored in the corresponding entry of attr_types.
    """
    def __init__(self, attrs, attr_types):
        assert "name" in attrs and "pose" in attrs
        for attr_name, arg in attrs.items():
            if attr_name == "pose" and arg == "undefined":
                self.pose = "undefined"
            else:
                setattr(self, attr_name, attr_types[attr_name](arg))

    def is_defined(self):
        return self.pose != "undefined"

class Symbol(Parameter):
    """
    Symbols must have at minimum a name and a value attribute (a d-by-T table, which we refer to as the
    trajectory table), which is set to "undefined" if the symbol's value is
    not defined. The attributes for the symbols are defined in the configuration files.
    attrs is a dictionary from instance attribute name to the arguments to pass into the __init__ method
    for the class stored in the corresponding entry of attr_types.
    """
    def __init__(self, attrs, attr_types):
        assert "name" in attrs and "value" in attrs
        for attr_name, arg in attrs.items():
            if attr_name == "value" and arg == "undefined":
                self.value = "undefined"
            else:
                setattr(self, attr_name, attr_types[attr_name](arg))

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

class Workspace(Parameter):
    def __init__(self, attrs, attr_types):
        for attr_name, arg in attrs.items():
            setattr(self, attr_name, attr_types[attr_name](arg))
