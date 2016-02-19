import unittest
from core.parsing import parse_domain_config
from errors_exceptions import DomainConfigException, PredicateException

class TestParseDomainConfig(unittest.TestCase):
    def setUp(self):
        self.c = {'Action moveto 20': '(?robot - Robot ?start - RPose ?end - RPose) (and (RobotAt ?robot ?start) (forall (?obj - Can) (not (Obstructs ?robot ?start ?obj)))) (and (not (RobotAt ?robot ?start)) (RobotAt ?robot ?end)) 0:0 0:19 19:19 19:19', 'Action putdown 20': '(?robot - Robot ?can - Can ?target - Target ?pdp - RPose) (and (RobotAt ?robot ?pdp) (IsPDP ?pdp ?target) (InGripper ?can) (forall (?obj - Can) (not (At ?obj ?target))) (forall (?obj - Can) (not (Obstructs ?robot ?pdp ?obj)))) (and (At ?can ?target) (not (InGripper ?can))) 0:0 0:0 0:0 0:0 0:19 19:19 19:19', 'Derived Predicates': 'At, Can, Target, RPose, Robot, Workspace', 'Attribute Import Paths': 'RedCircle core.util_classes.circle, BlueCircle core.util_classes.circle, GreenCircle core.util_classes.circle, Vector2d core.util_classes.matrix, GridWorldViewer core.util_classes.viewer', 'Primitive Predicates': 'geom, Can, RedCircle; pose, Can, Vector2d; geom, Target, BlueCircle; pose, Target, Vector2d; value, RPose, Vector2d; geom, Robot, GreenCircle; pose, Robot, Vector2d; pose, Workspace, Vector2d; w, Workspace, int; h, Workspace, int; size, Workspace, int; viewer, Workspace, GridWorldViewer', 'Action grasp 20': '(?robot - Robot ?can - Can ?target - Target ?gp - RPose) (and (At ?can ?target) (RobotAt ?robot ?gp) (IsGP ?gp ?can) (forall (?obj - Can) (not (InGripper ?obj))) (forall (?obj - Can) (not (Obstructs ?robot ?gp ?obj)))) (and (not (At ?can ?target)) (InGripper ?can) (forall (?sym - RPose) (not (Obstructs ?robot ?sym ?can)))) 0:0 0:0 0:0 0:0 0:19 19:19 19:19 19:19', 'Types': 'Can, Target, RPose, Robot, Workspace'}
        self.domain = parse_domain_config.ParseDomainConfig.parse(self.c)

    def test_param_schemas(self):
        s = self.domain.param_schemas
        self.assertEqual(len(s), 5)
        self.assertEqual(s["Can"].param_class.__name__, "Object")
        self.assertEqual(s["Target"].param_class.__name__, "Object")
        self.assertEqual(s["RPose"].param_class.__name__, "Symbol")
        self.assertEqual(s["Robot"].param_class.__name__, "Object")
        self.assertEqual(s["Workspace"].param_class.__name__, "Object")
        self.assertEqual(len(s["Can"].attr_dict), 4)
        self.assertEqual(s["Can"].attr_dict["name"], str)
        self.assertEqual(s["Can"].attr_dict["_type"], str)
        self.assertEqual(s["Can"].attr_dict["geom"](5.4).radius, 5.4)
        self.assertEqual(s["Can"].attr_dict["pose"]((3, 5))[0], 3)
        self.assertEqual(s["Workspace"].attr_dict["_type"], str)

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
        self.assertEqual(set(s.keys()), set(["At"]))
        self.assertEqual(s["At"].pred_class.__name__, "At")
        self.assertEqual(s["At"].expected_params, ["Can", "Target", "RPose", "Robot", "Workspace"])

    def test_pred_schemas_failure(self):
        new_c = self.c.copy()
        new_c["Derived Predicates"] = "Inside, Can, Target, RPose, Robot, Workspace"
        with self.assertRaises(PredicateException) as cm:
            parse_domain_config.ParseDomainConfig.parse(new_c)
        self.assertEqual(cm.exception.message, "Predicate type 'Inside' not defined!")

    def test_action_schemas_basic(self):
        s = self.domain.action_schemas["grasp"]
        self.assertEqual(s.name, "grasp")
        self.assertEqual(s.horizon, 20)
        self.assertEqual(s.params, [("?robot", "Robot"), ("?can", "Can"), ("?target", "Target"), ("?gp", "RPose")])
        self.assertEqual(s.universally_quantified_params, {"?obj": "Can", "?sym": "RPose", "?obj1": "Can"})
        self.assertTrue({"type": "Obstructs", "active_timesteps": (0, 19), "negated": True, "args": ["?robot", "?gp", "?obj1"]} in s.preds)
        s = self.domain.action_schemas["moveto"]
        self.assertEqual(s.name, "moveto")
        self.assertEqual(s.horizon, 20)
        self.assertEqual(s.params, [("?robot", "Robot"), ("?start", "RPose"), ("?end", "RPose")])
        self.assertTrue({"type": "RobotAt", "active_timesteps": (0, 0), "negated": False, "args": ["?robot", "?start"]} in s.preds)
        self.assertTrue({"type": "RobotAt", "active_timesteps": (19, 19), "negated": True, "args": ["?robot", "?start"]} in s.preds)
        self.assertTrue({"type": "RobotAt", "active_timesteps": (19, 19), "negated": False, "args": ["?robot", "?end"]} in s.preds)

    def test_action_schemas_nested_forall(self):
        new_c = self.c.copy()
        new_c["Action grasp 20"] = "(?robot - Robot) (and (forall (?sym - RPose) (RobotAt ?robot ?sym))) (and (forall (?obj - Can) (forall (?sym - RPose) (not (Obstructs ?robot ?sym ?obj))))) 0:0 0:19"
        s = parse_domain_config.ParseDomainConfig.parse(new_c).action_schemas["grasp"]
        self.assertEqual(s.params, [("?robot", "Robot")])
        self.assertEqual(s.universally_quantified_params, {"?obj": "Can", "?sym": "RPose", "?sym1": "RPose"})
        self.assertEqual(s.preds, [{"type": "RobotAt", "active_timesteps": (0, 0), "negated": False, "args": ["?robot", "?sym"]},
                                   {"type": "Obstructs", "active_timesteps": (0, 19), "negated": True, "args": ["?robot", "?sym1", "?obj"]}])

    def test_action_schemas_formatting(self):
        new_c = self.c.copy()
        new_c["Action grasp 20"] = "(?robot- Robot)(and (forall (?sym -RPose) (RobotAt ?robot ?sym ))) (and (  forall (?obj    - Can)    (forall(?sym-RPose)(not(Obstructs ?robot ?sym ?obj)))))    0:0  0:19"
        s = parse_domain_config.ParseDomainConfig.parse(new_c).action_schemas["grasp"]
        self.assertEqual(s.params, [("?robot", "Robot")])
        self.assertEqual(s.universally_quantified_params, {"?obj": "Can", "?sym": "RPose", "?sym1": "RPose"})
        self.assertEqual(s.preds, [{"type": "RobotAt", "active_timesteps": (0, 0), "negated": False, "args": ["?robot", "?sym"]},
                                   {"type": "Obstructs", "active_timesteps": (0, 19), "negated": True, "args": ["?robot", "?sym1", "?obj"]}])
