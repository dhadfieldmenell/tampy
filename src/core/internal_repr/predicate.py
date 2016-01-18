from IPython import embed as shell

class Predicate(object):
    """
    Predicates hold a set of parameters (see Parameter class) and represent testable relationships among
    these parameters. The test occurs for a particular time (0-indexed). A concrete predicate is one in which all
    the non-symbol parameters have values. Commonly used predicates can be found in the util_classes folder.
    """
    def __init__(self, name, params, expected_param_types):
        self.name = name
        self.params = params
        self.validate_params(expected_param_types)

    def get_type(self):
        return self.__class__.__name__

    def is_concrete(self):
        for param in self.params:
            if not param.is_symbol() and not param.is_defined():
                return False
        return True

    def test(self, time):
        if not self.is_concrete():
            return False
        raise NotImplementedError("Override this.")

    def validate_params(self, expected_param_types):
        if len(self.params) != len(expected_param_types):
            raise Exception("Parameter type validation failed for predicate '%s'."%self)
        for i, p in enumerate(self.params):
            if not p.get_type() == expected_param_types[i]:
                raise Exception("Parameter type validation failed for predicate '%s'."%self)

    def __repr__(self):
        s = "%s: (%s "%(self.name, self.get_type())
        for param in self.params[:-1]:
            s += param.name + " "
        s += self.params[-1].name + ")"
        return s
