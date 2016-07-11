import unittest
from core.parsing import parse_domain_config
from errors_exceptions import DomainConfigException, PredicateException

class TestParseDomainConfig(unittest.TestCase):
    def setUp(self):
        self.c = {
            'Types':'Can, Target, RobotPose, Robot, Grasp',
            'Attribute Import Paths':'RedCircle core.util_classes.circle, BlueCircle core.util_classes.circle, GreenCircle core.util_classes.circle, Vector2d core.util_classes.matrix, GridWorldViewer core.util_classes.viewer',
            'Predicates Import Path':'core.util_classes.namo_predicates',
            'Primitive Predicates':'geom, Can, RedCircle; pose, Can, Vector2d; geom, Target, BlueCircle; pose, Target, Vector2d; value, RobotPose, Vector2d; geom, Robot, GreenCircle; pose, Robot, Vector2d; value, Grasp, Vector2d',
            'Derived Predicates':'At, Can, Target; RobotAt, Robot, RobotPose; InGripper, Robot, Can, Grasp; InContact, Robot, RobotPose, Target; NotObstructs, Robot, RobotPose, Can; NotObstructsHolding, Robot, RobotPose, Can, Can; Stationary, Can; GraspValid, RobotPose, Target, Grasp',
            'Action moveto 20':'(?robot - Robot ?start - RobotPose ?end - RobotPose) (forall (?c-Can ?g-Grasp) (not (InGripper ?robot ?c ?g))) (and (RobotAt ?robot ?start) (forall (?obj - Can ?t - Target) (or (not (At ?obj ?t)) (not (NotObstructs ?robot ?end ?obj))))) (not (RobotAt ?robot ?start)) (RobotAt ?robot ?end) 0:0 0:0 0:19 19:19 19:19',
            'Action movetoholding 20':'(?robot - Robot ?start - RobotPose ?end - RobotPose ?c - Can ?g - Grasp) (RobotAt ?robot ?start) (InGripper ?robot ?c ?g) (forall (?obj - Can) (or (not (At ?obj ?t)) (not (NotObstructsHolding ?robot ?end ?obj ?c)))) (not (RobotAt ?robot ?start)) (RobotAt ?robot ?end) 0:0 0:19 0:19 19:19 19:19',
            'Action grasp 2':'(?robot - Robot ?can - Can ?target - Target ?gp - RobotPose ?g - Grasp) (and (At ?can ?target) (RobotAt ?robot ?gp) (InContact ?robot ?gp ?target) (GraspValid ?gp ?target ?g) (forall (?obj - Can ?g - Grasp) (not (InGripper ?robot ?obj ?g)))) (and (not (At ?can ?target)) (InGripper ?robot ?can ?g) (forall (?sym - RobotPose) (not (NotObstructs ?robot ?sym ?can))) (forall (?sym-Robotpose ?obj-Can) (not (NotObstructs ?robot ?sym ?can ?obj)))) 0:0 0:0 0:0 0:0 0:0 0:1 1:1 1:1 1:1',
            'Action putdown 2':'(?robot - Robot ?can - Can ?target - Target ?pdp - RobotPose ?g - Grasp) (and (RobotAt ?robot ?pdp) (InContact ?robot ?pdp ?target) (GraspValid ?pdp ?target ?g) (InGripper ?robot ?can ?g) (forall (?obj - Can) (not (At ?obj ?target))) (forall (?obj - Can) (not (NotObstructsHolding ?robot ?pdp ?obj ?can ?g)))) (and (At ?can ?target) (not (InGripper ?robot ?can ?g))) 0:0 0:0 0:0 0:0 0:0 0:1 1:1 1:1'}
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
        self.assertTrue({"type": "Obstructs", "hl_info": "pre", "active_timesteps": (0, 19), "negated": True, "args": ["?robot", "?gp", "?obj1"]} in s.preds)
        s = self.domain.action_schemas["moveto"]
        self.assertEqual(s.name, "moveto")
        self.assertEqual(s.horizon, 20)
        self.assertEqual(s.params, [("?robot", "Robot"), ("?start", "RPose"), ("?end", "RPose")])
        self.assertTrue({"type": "RobotAt", "hl_info": "pre", "active_timesteps": (0, 0), "negated": False, "args": ["?robot", "?start"]} in s.preds)
        self.assertTrue({"type": "RobotAt", "hl_info": "eff", "active_timesteps": (19, 19), "negated": True, "args": ["?robot", "?start"]} in s.preds)
        self.assertTrue({"type": "RobotAt", "hl_info": "eff", "active_timesteps": (19, 19), "negated": False, "args": ["?robot", "?end"]} in s.preds)

    def test_action_schemas_nested_forall(self):
        new_c = self.c.copy()
        new_c["Action grasp 20"] = "(?robot - Robot) (and (forall (?sym - RPose) (RobotAt ?robot ?sym))) (and (forall (?obj - Can) (forall (?sym - RPose) (not (Obstructs ?robot ?sym ?obj))))) 0:0 0:19"
        s = parse_domain_config.ParseDomainConfig.parse(new_c).action_schemas["grasp"]
        self.assertEqual(s.params, [("?robot", "Robot")])
        self.assertEqual(s.universally_quantified_params, {"?obj": "Can", "?sym": "RPose", "?sym1": "RPose"})
        self.assertEqual(s.preds, [{"type": "RobotAt", "hl_info": "pre", "active_timesteps": (0, 0), "negated": False, "args": ["?robot", "?sym"]},
                                   {"type": "Obstructs", "hl_info": "eff", "active_timesteps": (0, 19), "negated": True, "args": ["?robot", "?sym1", "?obj"]}])

    def test_action_schemas_formatting(self):
        new_c = self.c.copy()
        new_c["Action grasp 20"] = "(?robot- Robot)(and (forall (?sym -RPose) (RobotAt ?robot ?sym ))) (and (  forall (?obj    - Can)    (forall(?sym-RPose)(not(Obstructs ?robot ?sym ?obj)))))    0:0  0:19"
        s = parse_domain_config.ParseDomainConfig.parse(new_c).action_schemas["grasp"]
        self.assertEqual(s.params, [("?robot", "Robot")])
        self.assertEqual(s.universally_quantified_params, {"?obj": "Can", "?sym": "RPose", "?sym1": "RPose"})
        self.assertEqual(s.preds, [{"type": "RobotAt", "hl_info": "pre", "active_timesteps": (0, 0), "negated": False, "args": ["?robot", "?sym"]},
                                   {"type": "Obstructs", "hl_info": "eff", "active_timesteps": (0, 19), "negated": True, "args": ["?robot", "?sym1", "?obj"]}])


if __name__ == "__main__":
    unittest.main()
