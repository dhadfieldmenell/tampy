import unittest
from core.internal_repr import parameter_schema

class TestParameterSchema(unittest.TestCase):
    def test(self):
        s = parameter_schema.ParameterSchema(1, 2, 3)
        self.assertEqual(s.param_type, 1)
        self.assertEqual(s.param_class, 2)
        self.assertEqual(s.attr_dict, 3)
