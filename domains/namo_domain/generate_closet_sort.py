import itertools
import random

from core.util_classes.namo_predicates import dsafe

# NUM_CANS = 4


GOAL = "(RobotAt pr2 robot_end_pose)"
HEIGHT = 5
WIDTH = 5


def main():
    for NUM_CANS in range(1, 20):
        filename = "namo_probs/sort_closet_prob_{0}.prob".format(NUM_CANS)
        s = "# AUTOGENERATED. DO NOT EDIT.\n# Configuration file for NAMO problem instance. Blank lines and lines beginning with # are filtered out.\n\n"
        coords = list(itertools.product(range(-HEIGHT, HEIGHT), range(-WIDTH, WIDTH)))
        random.shuffle(coords)
        coord_ind = 0
        s += "# The values after each attribute name are the values that get passed into the __init__ method for that attribute's class defined in the domain configuration.\n"
        s += "Objects: "
        for n in range(NUM_CANS):
            s += "Target (name can%d_init_target); "%(n)
            s += "RobotPose (name pdp_target%d); "%(n)
            s += "Can (name can%d); "%(n)
            s += "Target (name can%d_end_target); "%(n)

        s += "Robot (name %s); "%"pr2"
        s += "Grasp (name {}); ".format("grasp0")
        s += "RobotPose (name %s); "%"robot_init_pose"
        s += "RobotPose (name %s); "%"robot_end_pose"
        s += "RobotPose (name %s); "%"grasp_end_pose"
        s += "Target (name %s); "%"middle_target"
        s += "Target (name %s); "%"left_target_1"
        s += "Target (name %s); "%"right_target_1"
        s += "Target (name %s); "%"left_target_2"
        s += "Target (name %s); "%"right_target_2"
        s += "Target (name %s); "%"aux_target_1"
        s += "Target (name %s); "%"aux_target_2"
        s += "Target (name %s); "%"aux_target_3"
        s += "Target (name %s); "%"aux_target_4"
        s += "Target (name %s); "%"aux_target_0"
        s += "Obstacle (name %s) \n\n"%"obs0"

        s += "Init: "
        for i in range(NUM_CANS):
            s += "(geom can%d_init_target 0.3), (value can%d_init_target %s), "%(i, i, list(coords[i]))
            s += "(value pdp_target%d undefined), "%i
            s += "(gripper pdp_target%d undefined), "%i
            s += "(geom can%d 0.3), (pose can%d %s), "%(i, i, list(coords[i]))
            s += "(geom can%d_end_target 0.3), (value can%d_end_target %s), "%(i, i, list(coords[i]))
        # s += "(value grasp0 undefined), "
        s += "(value grasp0 [0, {0}]), ".format(-0.6-dsafe)
        s += "(geom %s 0.3), (pose %s %s), "%("pr2", "pr2", [0, 0])
        s += "(gripper pr2 [0.]), "
        s += "(value %s %s), "%("robot_init_pose", [0., 0.])
        s += "(value %s %s), "%("robot_end_pose", "undefined")
        s += "(geom %s %s), "%("robot_init_pose", 0.3)
        s += "(geom %s %s), "%("robot_end_pose", 0.3)
        s += "(gripper %s [0.]), "%("robot_init_pose")
        s += "(gripper %s undefined), "%("robot_end_pose")
        s += "(value %s %s), "%("grasp_end_pose", "undefined")
        s += "(gripper %s %s), "%("grasp_end_pose", "undefined")
        s += "(value %s [0., 0.]), "%("middle_target")
        s += "(value %s [-1., 0.]), "%("left_target_1")
        s += "(value %s [1., 0.]), "%("right_target_1")
        s += "(value %s [-2., 0.]), "%("left_target_2")
        s += "(value %s [2., 0.]), "%("right_target_2")
        s += "(value %s [0., 0.]), "%("aux_target_1")
        s += "(value %s [0., 0.]), "%("aux_target_2")
        s += "(value %s [0., 0.]), "%("aux_target_3")
        s += "(value %s [0., 0.]), "%("aux_target_4")
        s += "(value %s [0., 0.]), "%("aux_target_0")
        s += "(pose %s [-3.5, 0]), "%"obs0"
        s += "(geom %s %s); "%("obs0", "closet")

        # for i in range(NUM_CANS):
            # s += "(At can{} can{}_init_target), ".format(i, i)
            # s += "(Stationary can{}), ".format(i)
            # for j in range(NUM_CANS):
            #     s += "(StationaryNEq can{} can{}), ".format(i, j)
            # s += "(InContact pr2 pdp_target{} can{}_init_target), ".format(i, i)
            # s += "(GraspValid pdp_target{} can{}_init_target grasp0), ".format(i, i)

        s += "(RobotAt pr2 robot_init_pose), "
        s += "(IsMP pr2), "
        s += "(StationaryW obs0) \n\n"

        s += "Goal: %s"%GOAL

        with open(filename, "w") as f:
            f.write(s)

if __name__ == "__main__":
    main()

