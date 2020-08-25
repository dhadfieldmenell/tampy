import unittest
from errors_exceptions import *

class TestExceptions(unittest.TestCase):
    def test_exception_classes(self):
        with self.assertRaises(ProblemConfigException):
            raise ProblemConfigException("A Problem Config Exception")
        with self.assertRaises(DomainConfigException):
            raise DomainConfigException("A Domain Config Exception")
        with self.assertRaises(SolversConfigException):
            raise SolversConfigException("A Solvers Config Exception")
        with self.assertRaises(ParamValidationException):
            raise ParamValidationException("A Param Validation Exception")
        with self.assertRaises(PredicateException):
            raise PredicateException("A Predicate Exception")
        with self.assertRaises(HLException):
            raise HLException("A High Level Solver Exception")
        with self.assertRaises(LLException):
            raise LLException("A Low Level Solver Exception")
        with self.assertRaises(ImpossibleException):
            raise ImpossibleException("An Impossible Exception")
