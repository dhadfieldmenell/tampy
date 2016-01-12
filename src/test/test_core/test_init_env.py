import unittest
from core import init_env
from core.internal_repr import predicate
from core.internal_repr import parameter
import numpy as np

class TestInitEnv(unittest.TestCase):
    def test_namo_init(self):
        params = [parameter.Target("target4"),
                  parameter.Target("target7"),
                  parameter.Target("target9"),
                  parameter.Robot("pr2"),
                  parameter.Can("can1"),
                  parameter.Can("can2"),
                  parameter.Can("can3")]
        preds = [predicate.At("at1", [params[-3], params[0]], ["Can", "Target"]),
                 predicate.At("at2", [params[-2], params[1]], ["Can", "Target"]),
                 predicate.At("at3", [params[-1], params[2]], ["Can", "Target"])]
        env_data = init_env.InitNAMOEnv().construct_env_and_init_params("namo_test.namo",
                                                                        {p.name:p for p in params},
                                                                        preds)
        self.assertEqual(env_data["w"], 9)
        self.assertEqual(env_data["h"], 10)
        self.assertTrue(np.array_equal(params[0].pose, [[1], [2]]))
        self.assertTrue(np.array_equal(params[1].pose, [[7], [7]]))
        self.assertTrue(np.array_equal(params[2].pose, [[8], [1]]))
        self.assertTrue(np.array_equal(params[3].pose, [[4], [2]]))
        self.assertTrue(np.array_equal(params[0].pose, params[-3].pose))
        self.assertTrue(np.array_equal(params[1].pose, params[-2].pose))
        self.assertTrue(np.array_equal(params[2].pose, params[-1].pose))
