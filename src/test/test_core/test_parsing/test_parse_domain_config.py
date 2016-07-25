import unittest
from core.parsing import parse_domain_config
from errors_exceptions import DomainConfigException, PredicateException
import main

class TestParseDomainConfig(unittest.TestCase):
    def setUp(self):
        domain_fname = '../domains/namo_domain/namo.domain'
        self.c = main.parse_file_to_dict(domain_fname)
        self.domain = parse_domain_config.ParseDomainConfig.parse(self.c)

    def test_param_schemas(self):
        s = self.domain.param_schemas
        self.assertEqual(len(s), 6)
        self.assertEqual(s["Can"].param_class.__name__, "Object")
        self.assertEqual(s["Target"].param_class.__name__, "Symbol")
        self.assertEqual(s["RobotPose"].param_class.__name__, "Symbol")
        self.assertEqual(s["Robot"].param_class.__name__, "Object")
        self.assertEqual(len(s["Can"].attr_dict), 4)
        self.assertEqual(s["Can"].attr_dict["name"], str)
        self.assertEqual(s["Can"].attr_dict["_type"], str)
        self.assertEqual(s["Can"].attr_dict["geom"](5.4).radius, 5.4)
        self.assertEqual(s["Can"].attr_dict["pose"]((3, 5))[0], 3)

    def test_param_schemas_failure(self):
        new_c = self.c.copy()
        new_c["Attribute Import Paths"] = "RedCircle core.util_classes.circle"
        with self.assertRaises(DomainConfigException) as cm:
            parse_domain_config.ParseDomainConfig.parse(new_c)
        self.assertEqual(cm.exception.message, "Need to provide attribute import path for non-primitive Vector2d.")

        del new_c["Attribute Import Paths"]
        with self.assertRaises(DomainConfigException) as cm:
            parse_domain_config.ParseDomainConfig.parse(new_c)
        self.assertTrue(cm.exception.message.startswith("Need to provide attribute import path for"))

        new_c["Attribute Import Paths"] = "RedCircle core.internal_repr, BlueCircle core.util_classes.circle, GreenCircle core.util_classes.circle, Vector2d core.util_classes.matrix, GridWorldViewer core.util_classes.viewer"
        with self.assertRaises(DomainConfigException) as cm:
            parse_domain_config.ParseDomainConfig.parse(new_c)
        self.assertTrue(cm.exception.message.startswith("RedCircle not found in module"))

    def test_pred_schemas(self):
        s = self.domain.pred_schemas
        self.assertEqual(set(s.keys()), set(["At", "Obstructs", "ObstructsHolding", "GraspValid", "InGripper", "RobotAt",
                                             "InContact", "Stationary", "StationaryNEq", "IsMP", "RCollides", "Collides", "StationaryW"]))
        self.assertEqual(s["At"].pred_class.__name__, "At")
        self.assertEqual(s["At"].expected_params, ["Can", "Target"])

    def test_pred_schemas_failure(self):
        new_c = self.c.copy()
        new_c["Derived Predicates"] = "Inside, Can, Target, RobotPose, Robot, Workspace"
        with self.assertRaises(PredicateException) as cm:
            parse_domain_config.ParseDomainConfig.parse(new_c)
        self.assertEqual(cm.exception.message, "Predicate type 'Inside' not defined!")

    def test_action_schemas_basic(self):
        s = self.domain.action_schemas["grasp"]
        self.assertEqual(s.name, "grasp")
        self.assertEqual(s.horizon, 20)
        self.assertEqual(s.params, [('?robot', 'Robot'), ('?can', 'Can'), ('?target', 'Target'), ('?sp', 'RobotPose'), ('?gp', 'RobotPose'), ('?g', 'Grasp')])
        self.assertEqual(s.universally_quantified_params, {'?obj_1': 'Can', '?obj_3': 'Can', '?obj_2': 'Can', '?obj_5': 'Can', '?obj_4': 'Can', '?w': 'Obstacle', '?sym2_1': 'RobotPose', '?sym2': 'RobotPose', '?sym1': 'RobotPose', '?obj': 'Can', '?w_2': 'Obstacle', '?w_1': 'Obstacle', '?g': 'Grasp', '?sym1_1': 'Robotpose'})

        s = self.domain.action_schemas["moveto"]
        self.assertEqual(s.name, "moveto")
        self.assertEqual(s.horizon, 20)
        self.assertEqual(s.params, [("?robot", "Robot"), ("?start", "RobotPose"), ("?end", "RobotPose")])
        self.assertTrue({"type": "RobotAt", "hl_info": "pre", "active_timesteps": (0, 0), "negated": False, "args": ["?robot", "?start"]} in s.preds)
        self.assertTrue({"type": "RobotAt", "hl_info": "eff", "active_timesteps": (19, 19), "negated": True, "args": ["?robot", "?start"]} in s.preds)
        self.assertTrue({"type": "RobotAt", "hl_info": "eff", "active_timesteps": (19, 19), "negated": False, "args": ["?robot", "?end"]} in s.preds)

    def test_action_schemas_nested_forall(self):
        new_c = self.c.copy()
        new_c["Action grasp 20"] = "(?robot - Robot) (and (forall (?sym - RobotPose) (RobotAt ?robot ?sym))) (and (forall (?obj - Can) (forall (?sym - RobotPose) (not (Obstructs ?robot ?sym ?obj))))) 0:0 0:19"
        s = parse_domain_config.ParseDomainConfig.parse(new_c).action_schemas["grasp"]
        self.assertEqual(s.params, [("?robot", "Robot")])
        self.assertEqual(s.universally_quantified_params, {"?obj": "Can", "?sym": "RobotPose", "?sym_1": "RobotPose"})
        self.assertEqual(s.preds, [{"type": "RobotAt", "hl_info": "pre", "active_timesteps": (0, 0), "negated": False, "args": ["?robot", "?sym"]},
                                   {"type": "Obstructs", "hl_info": "eff", "active_timesteps": (0, 19), "negated": True, "args": ["?robot", "?sym_1", "?obj"]}])

    def test_action_schemas_formatting(self):
        new_c = self.c.copy()
        new_c["Action grasp 20"] = "(?robot- Robot)(and (forall (?sym -RobotPose) (RobotAt ?robot ?sym ))) (and (  forall (?obj    - Can)    (forall(?sym-RobotPose)(not(Obstructs ?robot ?sym ?obj)))))    0:0  0:19"
        s = parse_domain_config.ParseDomainConfig.parse(new_c).action_schemas["grasp"]
        self.assertEqual(s.params, [("?robot", "Robot")])
        self.assertEqual(s.universally_quantified_params, {"?obj": "Can", "?sym": "RobotPose", "?sym_1": "RobotPose"})
        self.assertEqual(s.preds, [{"type": "RobotAt", "hl_info": "pre", "active_timesteps": (0, 0), "negated": False, "args": ["?robot", "?sym"]},
                                   {"type": "Obstructs", "hl_info": "eff", "active_timesteps": (0, 19), "negated": True, "args": ["?robot", "?sym_1", "?obj"]}])


if __name__ == "__main__":
    unittest.main()
