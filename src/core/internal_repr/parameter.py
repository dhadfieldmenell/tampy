from IPython import embed as shell
from errors_exceptions import DomainConfigException
from core.util_classes.matrix import Vector
from core.util_classes.openrave_body import OpenRAVEBody

import h5py
import numpy as np

class Parameter(object):
    """
    Parameters fall into one of two categories: Objects and Symbols. Objects are
    things that exist in the environment with some pose, and Symbols are
    symbolic references. To store information about the environment that's
    useful for planning, we often spawn a workspace Object (but this is not
    necessary). Objects and Symbols are distinguished in the config file parsing
    as follows: Objects have a "pose" instance attribute while Symbols have a
    "value" instance attribute.
    """
    def __init__(self, attrs=None, attr_types=None):
        self.openrave_body = None
        self._free_attrs = {}
        self._saved_free_attrs = {}

        if attr_types is not None:
            self._attr_types = attr_types.copy()
            self._attr_types['_attr_types'] = dict
        else:
            self._attr_types = {'attr_types': dict}

        if attrs is not None:
            for attr_name, arg in attrs.items():
                if "undefined" in arg:
                    setattr(self, attr_name, "undefined")
                else:
                    try:
                        setattr(self, attr_name, attr_types[attr_name](*arg))
                    except KeyError:
                        name = attrs["name"][0]
                        raise DomainConfigException("Attribute '{}' for {} '{}' not defined in domain file.".format(attr_name, type(self).__name__, name))

    def get_attr_type(self, attr_name):
        raise NotImplementedError("get_attr_type not implemented for Parameter.")

    def get_attr_type(self, attr_name):
        if attr_name == 'openrave_body':
            return OpenRAVEBody
        elif attr_name == '_free_attrs':
            return dict
        elif attr_name == '_saved_free_attrs':
            return dict
        return self._attr_types[attr_name]

    def get_type(self):
        return self._type

    def is_symbol(self):
        return False

    def is_fixed(self, attr_list, t=None):
        if t is None:
            return not np.all([np.all(self._free_attrs[attr]) for attr in attr_list])
        else:
            return not np.all([self._free_attrs[attr][:, t] for attr in attr_list])

    def is_defined(self):
        for attr_name in self._attr_types.iterkeys():
            if getattr(self, attr_name) is "undefined":
                return False
        return True

    def save_free_attrs(self):
        self._saved_free_attrs = {}
        for k, v in self._free_attrs.iteritems():
            self._saved_free_attrs[k] = v.copy()

    def restore_free_attrs(self):
        self._free_attrs = self._saved_free_attrs


    def __repr__(self):
        return "%s - %s"%(self.name, self.get_type())

class Object(Parameter):
    """
    Objects must have at minimum a name, a type (a string), and a pose attribute
    (a d-by-T table, which we refer to as the trajectory table), which is set to
    "undefined" if the object's pose is not defined. The attributes for the
    objects are defined in the configuration files. attrs is a dictionary from
    instance attribute name to the argument to pass into the __init__ method for
    the class stored in the corresponding entry of attr_types. attrs must have,
    at minimum, the keys "name", "_type", and "pose".
    """
    def __init__(self, attrs=None, attr_types=None):
        if attrs is not None:
            assert "name" in attrs and "_type" in attrs and "pose" in attrs
        super(Object, self).__init__(attrs=attrs, attr_types=attr_types)

    def is_defined(self):
        return self.pose is not "undefined"

    def copy(self, new_horizon):
        new = Object()
        for attr_name, v in self.__dict__.items():
            attr_type = self.get_attr_type(attr_name)
            if issubclass(attr_type, Vector):
                new_value = np.empty((attr_type.dim, new_horizon))
                new_value[:] = np.NaN
                if v is not "undefined":
                    assert attr_type.dim == v.shape[0]
                    new_value[:v.shape[0], :v.shape[1]] = v[:v.shape[0], :min(v.shape[1], new_horizon)]
                setattr(new, attr_name, new_value)
            else:
                setattr(new, attr_name, v)
        return new

    def write_to_hdf5(self, file_name):
        hdf5_file = h5py.File(file_name, 'w')
        group = hdf5_file.create_group('trajectory')
        for attr_name, value in self.__dict__.items():
            if issubclass(self.get_attr_type(attr_name), Vector):
                group.create_dataset(attr_name, data=value)
        hdf5_file.close()

class Symbol(Parameter):
    """
    Symbols must have at minimum a name, a type (a string), and a value
    attribute, which is set to "undefined" if the symbol's value is not defined.
    Symbols are static, which means that it's value does not depend on time.
    The attributes for the symbols are defined in the configuration files. attrs
    is a dictionary from instance attribute name to the arguments to pass into
    the __init__ method for the class stored in the corresponding entry of
    attr_types. attrs must have, at minimum, the keys "name", "_type", and
    "value".
    """
    def __init__(self, attrs=None, attr_types=None):
        if attrs is not None:
            assert "name" in attrs and "_type" in attrs and "value" in attrs
        super(Symbol, self).__init__(attrs=attrs, attr_types=attr_types)

    def is_defined(self):
        return self.value is not "undefined"

    def is_symbol(self):
        return True

    def copy(self, new_horizon):
        new = Symbol()
        for k, v in self.__dict__.items():
            if v == 'undefined':
                attr_type = self.get_attr_type(k)
                assert issubclass(attr_type, Vector)
                val = np.empty((attr_type.dim, 1))
                val[:] = np.NaN
                setattr(new, k, val)
            else:
                setattr(new, k, v)
        return new
