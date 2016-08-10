from IPython import embed as shell
from errors_exceptions import ParamValidationException

class Predicate(object):
    """
    Predicates hold a set of parameters (see Parameter class) and represent testable relationships among
    these parameters. The test occurs for a particular time (0-indexed). A concrete predicate is one in which all
    the non-symbol parameters have values. Commonly used predicates can be found in the core/util_classes/ folder.
    """
    def __init__(self, name, params, expected_param_types, env=None, active_range=(0,0)):
        self.name = name
        self.params = params
        self.expected_param_types = expected_param_types
        self.validate_params(expected_param_types)
        self._env = env
        self.active_range = active_range
        self.priority = 0

    def get_type(self):
        return self.__class__.__name__

    def is_concrete(self):
        for param in self.params:
            if not param.is_defined():
                return False
        return True

    def test(self, time, negated=False):
        if not self.is_concrete():
            return False
        raise NotImplementedError("Override this.")

    def resample(self, negated, time, plan):
        return None, None

    def _get_param_copy(self, param_to_copy):
        return [param_to_copy[param] for param in self.params]

    def copy(self, param_to_copy):
        params = self._get_param_copy(param_to_copy)
        return Predicate(self.name, params, self.expected_param_types[:],
                         env=self._env, active_range=self.active_range)

    def validate_params(self, expected_param_types):
        if len(self.params) != len(expected_param_types):
            raise ParamValidationException("Parameter type validation failed for predicate '%s'."%self)
        for i, p in enumerate(self.params):
            if not p.get_type() == expected_param_types[i]:
                raise ParamValidationException("Parameter type validation failed for predicate '%s'."%self)

    def __repr__(self):
        s = "%s: (%s "%(self.name, self.get_type())
        for param in self.params[:-1]:
            s += param.name + " "
        s += self.params[-1].name + ")"
        return s
