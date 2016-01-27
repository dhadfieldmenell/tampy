import unittest
from core.parsing import parse_solvers_config
from errors_exceptions import SolversConfigException, HLException, LLException

class TestParseSolversConfig(unittest.TestCase):
    def test(self):
        config = {"HLSolver": "DummyHLSolver", "LLSolver": "DummyLLSolver"}
        hls, lls = parse_solvers_config.ParseSolversConfig.parse(config, None)
        self.assertEqual(hls.abs_domain, "translate domain")
        self.assertEqual(hls.translate_problem(None), "translate problem")
        self.assertEqual(hls.solve(None, None, None), "solve")
        self.assertEqual(lls.solve(None), "solve")

    def test_failures(self):
        config = {"HLSolver": "HLSolver", "LLSolver": "LLSolver"}
        with self.assertRaises(NotImplementedError) as cm:
            parse_solvers_config.ParseSolversConfig.parse(config, None)
        self.assertEqual(cm.exception.message, "Override this.")

        config = {"LLSolver": "DummyLLSolver"}
        with self.assertRaises(SolversConfigException) as cm:
            parse_solvers_config.ParseSolversConfig.parse(config, None)
        self.assertEqual(cm.exception.message, "Must define both HL solver and LL solver in solvers config file.")

        config = {"HLSolver": "DummyHLSolver"}
        with self.assertRaises(SolversConfigException) as cm:
            parse_solvers_config.ParseSolversConfig.parse(config, None)
        self.assertEqual(cm.exception.message, "Must define both HL solver and LL solver in solvers config file.")

        config = {"HLSolver": "foo", "LLSolver": "DummyLLSolver"}
        with self.assertRaises(HLException) as cm:
            parse_solvers_config.ParseSolversConfig.parse(config, None)
        self.assertEqual(cm.exception.message, "HLSolver 'foo' not defined!")

        config = {"HLSolver": "DummyHLSolver", "LLSolver": "bar"}
        with self.assertRaises(LLException) as cm:
            parse_solvers_config.ParseSolversConfig.parse(config, None)
        self.assertEqual(cm.exception.message, "LLSolver 'bar' not defined!")
