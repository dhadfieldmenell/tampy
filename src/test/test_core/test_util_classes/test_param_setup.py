import unittest
from core.util_classes.param_setup import ParamSetup
import numpy as np
class TestParamSetup(unittest.TestCase):

    def test_setup(self):
        env = ParamSetup.setup_env()
        pr2 = ParamSetup.setup_pr2()
        self.assertEqual(pr2.name, "pr2")
        pr2_pose = ParamSetup.setup_pr2_pose()
        self.assertEqual(pr2_pose.name, "pr2_pose")
        blue_can = ParamSetup.setup_blue_can()
        self.assertEqual(blue_can.name, "blue_can")
        red_can = ParamSetup.setup_red_can()
        self.assertEqual(red_can.name, "red_can")
        green_can = ParamSetup.setup_green_can()
        self.assertEqual(green_can.name, "green_can")
        target = ParamSetup.setup_target()
        self.assertEqual(target.name, "target")
        ee_pose = ParamSetup.setup_ee_pose()
        self.assertEqual(ee_pose.name, "ee_pose")
        table = ParamSetup.setup_table()
        self.assertEqual(table.name, "table")
        box = ParamSetup.setup_box()
        self.assertEqual(box.name, "box")
