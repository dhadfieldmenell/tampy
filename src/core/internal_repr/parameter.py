from IPython import embed as shell
from errors_exceptions import DomainConfigException
import numpy as np

class Parameter(object):
    """
    Parameters fall into one of two categories: Objects and Symbols. Objects are things that
    exist in the environment with some pose, and Symbols are symbolic references.
    To store information about the environment that's useful for planning, we often spawn a workspace
    Object (but this is not necessary). Objects and Symbols are distinguished in the config file parsing
    as follows: Objects have a "pose" instance attribute while Symbols have a "value" instance attribute.
    """
    def __init__(self, *args):
        raise NotImplementedError("Must instantiate either Object or Symbol.")

    def get_type(self):
        return self._type

    def is_symbol(self):
        return False

    def __repr__(self):
        return "%s - %s"%(self.name, self.get_type())

class Object(Parameter):
    """
    Objects must have at minimum a name, a type (a string), and a pose attribute (a d-by-T table, which we refer to as the
    trajectory table), which is set to "undefined" if the object's pose is
    not defined. The attributes for the objects are defined in the configuration files.
    attrs is a dictionary from instance attribute name to the argument to pass into the __init__ method
    for the class stored in the corresponding entry of attr_types. attrs must have, at minimum, the keys "name", "_type", and "pose".
    """
    def __init__(self, attrs=None, attr_types=None):
        if attrs is not None:
            assert "name" in attrs and "_type" in attrs and "pose" in attrs
            for attr_name, arg in attrs.items():
                if attr_name == "pose" and arg == "undefined":
                    self.pose = "undefined"
                else:
                    try:
                        setattr(self, attr_name, attr_types[attr_name](arg))
                    except KeyError:
                        raise DomainConfigException("Attribute '%s' for Object '%s' not defined in domain file."%(attr_name, attrs["name"]))

    def is_defined(self):
        return self.pose != "undefined"

    def copy(self, new_horizon):
        new = Object()
        for k, v in self.__dict__.items():
            if k == "pose" and self.is_defined():
                new.pose = np.zeros((v.shape[0], new_horizon))
                new.pose[:v.shape[0], :v.shape[1]] = v[:v.shape[0], :min(v.shape[1], new_horizon)]
            else:
                setattr(new, k, v)
        return new

class Symbol(Parameter):
    """
    Symbols must have at minimum a name, a type (a string), and a value attribute (a d-by-T table, which we refer to as the
    trajectory table), which is set to "undefined" if the symbol's value is
    not defined. The attributes for the symbols are defined in the configuration files.
    attrs is a dictionary from instance attribute name to the arguments to pass into the __init__ method
    for the class stored in the corresponding entry of attr_types. attrs must have, at minimum, the keys "name", "_type", and "value".
    """
    def __init__(self, attrs=None, attr_types=None):
        if attrs is not None:
            assert "name" in attrs and "_type" in attrs and "value" in attrs
            for attr_name, arg in attrs.items():
                if attr_name == "value" and arg == "undefined":
                    self.value = "undefined"
                else:
                    try:
                        setattr(self, attr_name, attr_types[attr_name](arg))
                    except KeyError:
                        raise DomainConfigException("Attribute '%s' for Symbol '%s' not defined in domain file."%(attr_name, attrs["name"]))

    def is_defined(self):
        return self.value != "undefined"

    def is_symbol(self):
        return True

    def copy(self, new_horizon):
        new = Symbol()
        for k, v in self.__dict__.items():
            if k == "value" and self.is_defined():
                new.value = np.zeros((v.shape[0], new_horizon))
                new.value[:v.shape[0], :v.shape[1]] = v[:v.shape[0], :min(v.shape[1], new_horizon)]
            else:
                setattr(new, k, v)
        return new
