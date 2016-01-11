import unittest
from core.internal_repr import parameter

class TestParameter(unittest.TestCase):
  def test_object(self):
      can = parameter.Can("can1")
      self.assertEqual(can.name, "can1")
      self.assertFalse(can.is_symbol())
      self.assertFalse(can.is_defined())
      can.pose = [3, 3]
      self.assertFalse(can.is_symbol())
      self.assertTrue(can.is_defined())
      self.assertEqual(can.get_type(), "Can")
      self.assertTrue(isinstance(can, parameter.Object))
      self.assertTrue(isinstance(can, parameter.Parameter))

  def test_symbol(self):
      sym = parameter.Symbol("sym1")
      self.assertEqual(sym.name, "sym1")
      self.assertTrue(sym.is_symbol())
      self.assertFalse(sym.is_defined())
      sym.value = [3, 3]
      self.assertTrue(sym.is_symbol())
      self.assertTrue(sym.is_defined())
      self.assertEqual(sym.get_type(), "Symbol")
      self.assertFalse(isinstance(sym, parameter.Object))
      self.assertTrue(isinstance(sym, parameter.Parameter))

  def test_errors(self):
      # cannot instantiate Parameter directly
      with self.assertRaises(NotImplementedError) as cm:
          parameter.Parameter("can1")
      self.assertEqual(cm.exception.message, "Must instantiate either Object or Symbol.")
      with self.assertRaises(NotImplementedError) as cm:
          parameter.Parameter("can2", 4, 76, 1)
      self.assertEqual(cm.exception.message, "Must instantiate either Object or Symbol.")
      with self.assertRaises(NotImplementedError) as cm:
          parameter.Parameter()
      self.assertEqual(cm.exception.message, "Must instantiate either Object or Symbol.")
