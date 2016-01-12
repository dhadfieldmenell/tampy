import unittest
from pma import parse_config_to_solvers

class TestParseConfigToSolvers(unittest.TestCase):
    def test(self):
        config = {"HLSolver": "DummyHLSolver", "LLSolver": "DummyLLSolver"}
        hls, lls = parse_config_to_solvers.ParseConfigToSolvers(config).parse()
        self.assertEqual(hls.translate(None, None), "translate")
        self.assertEqual(hls.solve(None, None), "solve")
        self.assertEqual(lls.solve(None), "solve")

    def test_failures(self):
        config = {"HLSolver": "HLSolver", "LLSolver": "LLSolver"}
        hls, lls = parse_config_to_solvers.ParseConfigToSolvers(config).parse()
        with self.assertRaises(NotImplementedError) as cm:
            hls.solve(None, None)
        self.assertEqual(cm.exception.message, "Override this.")

        config = {"LLSolver": "DummyLLSolver"}
        with self.assertRaises(Exception) as cm:
            parse_config_to_solvers.ParseConfigToSolvers(config).parse()
        self.assertEqual(cm.exception.message, "Must define both HL solver and LL solver in config file.")

        config = {"HLSolver": "DummyHLSolver"}
        with self.assertRaises(Exception) as cm:
            parse_config_to_solvers.ParseConfigToSolvers(config).parse()
        self.assertEqual(cm.exception.message, "Must define both HL solver and LL solver in config file.")

        config = {"HLSolver": "foo", "LLSolver": "DummyLLSolver"}
        with self.assertRaises(Exception) as cm:
            parse_config_to_solvers.ParseConfigToSolvers(config).parse()
        self.assertEqual(cm.exception.message, "HLSolver 'foo' not defined!")

        config = {"HLSolver": "DummyHLSolver", "LLSolver": "bar"}
        with self.assertRaises(Exception) as cm:
            parse_config_to_solvers.ParseConfigToSolvers(config).parse()
        self.assertEqual(cm.exception.message, "LLSolver 'bar' not defined!")
