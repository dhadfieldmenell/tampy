from IPython import embed as shell
import numpy as np

class Predicate:
    """
    Predicates hold a set of parameters (see Parameter class) and represent testable relationships among
    these parameters. The test occurs for a particular time slice (0-indexed). A concrete predicate is one in which all
    the non-symbol parameters have values.
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

    def test(self, start_time, end_time):
        if not self.is_concrete():
            return False
        raise NotImplementedError("Override this.")

    def validate_params(self, expected_param_types):
        if len(self.params) != len(expected_param_types):
            raise Exception("Parameter type validation failed for predicate '%s'."%self.name)
        for i, p in enumerate(self.params):
            if not p.get_type() == expected_param_types[i]:
                raise Exception("Parameter type validation failed for predicate '%s'."%self.name)

    def __repr__(self):
        s = "(%s "%self.get_type()
        for param in self.params[:-1]:
            s += param.name + " "
        s += self.params[-1].name + ")"
        return s

class At(Predicate):
    def test(self, start_time, end_time):
        if not self.is_concrete():
            return False
        # verify start and end times are valid
        T = self.params[0].pose.shape[1]
        if start_time > end_time or start_time < 0 or end_time > T - 1:
            raise Exception("Out of range start or end time for predicate '%s'."%self.name)
        return np.array_equal(self.params[0].pose[:, start_time:end_time+1], self.params[1].pose[:, start_time:end_time+1])

class RobotAt(Predicate):
    def test(self, start_time, end_time):
        return True

class IsGP(Predicate):
    def test(self, start_time, end_time):
        return True

class IsPDP(Predicate):
    def test(self, start_time, end_time):
        return True

class InGripper(Predicate):
    def test(self, start_time, end_time):
        # TODO
        return False

class Obstructs(Predicate):
    def test(self, start_time, end_time):
        # TODO
        return True
