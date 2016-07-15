import unittest
from core.internal_repr import plan
from core.internal_repr import parameter
from core.util_classes import circle
from core.util_classes.matrix import Vector2d

class TestPlan(unittest.TestCase):
    def test_args(self):
        attrs = {"name": ["robot"], "geom": [1], "pose": [(0,0)], "_type": ["Robot"]}
        attr_types = {"name": str, "geom": circle.RedCircle,"pose": Vector2d, "_type": str}
        robot = parameter.Object(attrs, attr_types)
        attrs = {"name": ["can"], "geom": [1], "pose": [(3,4)], "_type": ["Can"]}
        attr_types = {"name": str, "geom": circle.RedCircle, "pose": Vector2d, "_type": str}
        can = parameter.Object(attrs, attr_types)
        param_map = {"robot": robot, "can": can}
        act_list = ["up", "down"]

        s = plan.Plan(param_map, act_list, 3, 4)
        self.assertEqual(s.params, param_map)
        self.assertEqual(s.actions, act_list)
        self.assertEqual(s.horizon, 3)
        self.assertEqual(s.env, 4)

    # TODO: add more as functionality develops
if __name__ == "__main__":
    unittest.main()
