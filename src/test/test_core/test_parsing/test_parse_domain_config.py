import unittest
from core.parsing import parse_domain_config

class TestParseDomainConfig(unittest.TestCase):
    def setUp(self):
        self.c = {'Action moveto 20': '(?robot - Robot ?start - RPose ?end - RPose) (and (RobotAt ?robot ?start) (forall (?obj - Can) (not (Obstructs ?robot ?start ?obj)))) (and (not (RobotAt ?robot ?start)) (RobotAt ?robot ?end)) 0:0 0:19 19:19 19:19', 'Action putdown 20': '(?robot - Robot ?can - Can ?target - Target ?pdp - RPose) (and (RobotAt ?robot ?pdp) (IsPDP ?pdp ?target) (InGripper ?can) (forall (?obj - Can) (not (At ?obj ?target))) (forall (?obj - Can) (not (Obstructs ?robot ?pdp ?obj)))) (and (At ?can ?target) (not (InGripper ?can))) 0:0 0:0 0:0 0:0 0:19 19:19 19:19', 'Action grasp 20': '(?robot - Robot ?can - Can ?target - Target ?gp - RPose) (and (At ?can ?target) (RobotAt ?robot ?gp) (IsGP ?gp ?can) (forall (?obj - Can) (not (InGripper ?obj))) (forall (?obj - Can) (not (Obstructs ?robot ?gp ?obj)))) (and (not (At ?can ?target)) (InGripper ?can) (forall (?sym - RPose) (not (Obstructs ?robot ?sym ?can)))) 0:0 0:0 0:0 0:0 0:19 19:19 19:19 19:19', 'Attribute Import Paths': 'RedCircle core.util_classes.circle, BlueCircle core.util_classes.circle, GreenCircle core.util_classes.circle, Vector2d core.util_classes.matrix, GridWorldViewer core.util_classes.viewer', 'Predicates': 'At, Can, Target, RPose, Robot, Workspace', 'Types': 'Can (name str. geom RedCircle. pose Vector2d); Target (name str. pose Vector2d); RPose (name str. value Vector2d); Robot (name str. pose Vector2d); Workspace (name str. pose Vector2d)'}
        self.domain = parse_domain_config.ParseDomainConfig.parse(self.c)

    def test_param_schema(self):
        s = self.domain.param_schema
        self.assertEqual(len(s), 5)
        self.assertEqual(s["Can"][0].__name__, "Object")
        self.assertEqual(s["Target"][0].__name__, "Object")
        self.assertEqual(s["RPose"][0].__name__, "Symbol")
        self.assertEqual(s["Robot"][0].__name__, "Object")
        self.assertEqual(s["Workspace"][0].__name__, "Object")
        self.assertEqual(len(s["Can"][1]), 4)
        self.assertEqual(s["Can"][1]["name"], str)
        self.assertEqual(s["Can"][1]["_type"], str)
        self.assertEqual(s["Can"][1]["geom"](5.4).radius, 5.4)
        self.assertEqual(s["Can"][1]["pose"]((3, 5))[0], 3)
        self.assertEqual(s["Workspace"][1]["_type"], str)

    def test_param_schema_failure(self):
        new_c = self.c.copy()
        new_c["Attribute Import Paths"] = "RedCircle core.util_classes.circle"
        with self.assertRaises(Exception) as cm:
            parse_domain_config.ParseDomainConfig.parse(new_c)
        self.assertEqual(cm.exception.message, "Need to provide attribute import path for non-primitive Vector2d.")

        del new_c["Attribute Import Paths"]
        with self.assertRaises(Exception) as cm:
            parse_domain_config.ParseDomainConfig.parse(new_c)
        self.assertTrue(cm.exception.message.startswith("Need to provide attribute import path for"))

        new_c["Attribute Import Paths"] = "RedCircle core.internal_repr, BlueCircle core.util_classes.circle, GreenCircle core.util_classes.circle, Vector2d core.util_classes.matrix, GridWorldViewer core.util_classes.viewer"
        with self.assertRaises(Exception) as cm:
            parse_domain_config.ParseDomainConfig.parse(new_c)
        self.assertTrue(cm.exception.message.startswith("RedCircle not found in module"))

    def test_pred_schema(self):
        s = self.domain.pred_schema
        self.assertEqual(set(s.keys()), set(["At"]))
        self.assertEqual(s["At"][0].__name__, "At")
        self.assertEqual(s["At"][1], ["Can", "Target", "RPose", "Robot", "Workspace"])

    def test_pred_schema_failure(self):
        new_c = self.c.copy()
        new_c["Predicates"] = "Inside, Can, Target, RPose, Robot, Workspace"
        with self.assertRaises(Exception) as cm:
            parse_domain_config.ParseDomainConfig.parse(new_c)
        self.assertEqual(cm.exception.message, "Predicate type 'Inside' not defined!")

    def test_action_schema(self):
        # TODO
        pass
