from IPython import embed as shell
from core.internal_repr.predicate import Predicate
from errors_exceptions import PredicateException
import numpy as np

"""
This file implements the classes for commonly used predicates that are useful in a wide variety of
typical domains.
"""

class At(Predicate):
    def test(self, time):
        if not self.is_concrete():
            return False
        # verify time is valid
        T = self.params[0].pose.shape[1]
        if time < 0 or time > T - 1:
            raise PredicateException("Out of range time for predicate '%s'."%self)
        return np.array_equal(self.params[0].pose[:, time], self.params[1].pose[:, time])

class RobotAt(Predicate):
    def test(self, time):
        return True

class IsGP(Predicate):
    def test(self, time):
        return True

class IsPDP(Predicate):
    def test(self, time):
        return True

class InGripper(Predicate):
    def test(self, time):
        # TODO
        return False

class Obstructs(Predicate):
    def test(self, time):
        # TODO
        return True
