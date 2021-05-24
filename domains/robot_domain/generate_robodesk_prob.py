from IPython import embed as shell
import itertools
import numpy as np
import random


# SEED = 1234
NUM_PROBS = 1
filename = "probs/robodesk_prob.prob"
GOAL = "(RobotAt panda robot_end_pose)"


PANDA_INIT_POSE = [0., 0.1, 0.55]
PANDA_INIT_ROT = [0, 0, 1.57]
PANDA_END_POSE = [0., 0.1, 0.55]
R_ARM_INIT = [-0.30, -0.4, 0.28, -2.5, 0.13, 1.87, 0.91]
OPEN_GRIPPER = [0.04, 0.04]
CLOSE_GRIPPER = [0., 0.]
EE_POS = [0.11338, -0.16325, 1.03655]
EE_ROT = [3.139, 0.00, -2.182]

SHELF_TARGET_POS = [0.3, 1.2, 0.85]
SHELF_TARGET_ROT = [1.57, 0., 0.]
BIN_TARGET_POS = [0.4, 0.55, 0.75]
BIN_TARGET_ROT = [0., 0., 0.]
OFF_DESK_TARGET_POS = [0.8, 0.675, 0.75]
OFF_DESK_TARGET_ROT = [0., 0., 0.]

SHELF_GEOM = 'desk_shelf'
SHELF_POS = [0., 0.85, 0.]
SHELF_ROT = [0., 0., 0.]

DESK_BODY_GEOM = [0.6, 0.275, 0.025]
DESK_BODY_POS = [0., 0.85, 0.735]
DESK_BODY_ROT = [0, 0, 0]

DRAWER_GEOM = 'desk_drawer'
DRAWER_POS = [0., 0.85, 0.655]
DRAWER_ROT = [0. ,0., 0.]

def get_panda_pose_str(name, RArm = R_ARM_INIT, G = OPEN_GRIPPER, Pos = PANDA_INIT_POSE, Rot = PANDA_INIT_ROT):
    s = ""
    s += "(right {} {}), ".format(name, RArm)
    s += "(right_ee_pos {} {}), ".format(name, EE_POS)
    s += "(right_ee_rot {} {}), ".format(name, EE_ROT)
    s += "(right_gripper {} {}), ".format(name, G)
    s += "(value {} {}), ".format(name, Pos)
    s += "(rotation {} {}), ".format(name, Rot)
    return s

def get_panda_str(name, RArm = R_ARM_INIT, G = OPEN_GRIPPER, Pos = PANDA_INIT_POSE, Rot = PANDA_INIT_ROT):
    s = ""
    s += "(geom {})".format(name)
    s += "(right {} {}), ".format(name, RArm)
    s += "(right_ee_pos {} {}), ".format(name, EE_POS)
    s += "(right_ee_rot {} {}), ".format(name, EE_ROT)
    s += "(right_gripper {} {}), ".format(name, G)
    s += "(pose {} {}), ".format(name, Pos)
    s += "(rotation {} {}), ".format(name, Rot)
    return s

def get_undefined_robot_pose_str(name):
    s = ""
    s += "(right {} undefined), ".format(name)
    s += "(right_ee_pos {} undefined), ".format(name)
    s += "(right_ee_rot {} undefined), ".format(name)
    s += "(right_gripper {} undefined), ".format(name)
    s += "(value {} undefined), ".format(name)
    s += "(rotation {} undefined), ".format(name)
    return s

def get_undefined_symbol(name):
    s = ""
    s += "(value {} undefined), ".format(name)
    s += "(rotation {} undefined), ".format(name)
    return s

def main():
    s = "# AUTOGENERATED. DO NOT EDIT.\n# Configuration file for CAN problem instance. Blank lines and lines beginning with # are filtered out.\n\n"

    s += "# The values after each attribute name are the values that get passed into the __init__ method for that attribute's class defined in the domain configuration.\n"
    s += "Objects: "
    s += "Panda (name panda); "

    items = ['upright_block', 'flat_block', 'ball', 'red_button', 'green_button', 'blue_button', \
             'drawer_handle', 'shelf_handle']
    init_pos = [[0.15, 0.78, 0.85], [0.15, 0.63, 0.775], [-0.4, 0.7, 0.799], [-0.45, 1.0, 1.1], \
                [-0.25, 1.0, 1.1], [-0.05, 1.0, 1.1], [-0.3, 0.92, 0.935], [0., 0.814, 0.656]]
    dims = [[0.023, 0.09, 0.023], [0.08, 0.035, 0.015], [0.04], [0.035, 0.01], [0.035, 0.01], \
            [0.035, 0.01], [0.01, 0.05], [0.01, 0.05]]
    item_types = []
    for item in items:
        if item.find('block') >= 0:
            item_type = 'Box'
        elif item.find('ball') >= 0:
            item_type = 'Sphere'
        else:
            item_type = 'Can'
        item_types.append(item_type)

    for ind, item in enumerate(items):
        s += "PandaPose (name {}); ".format("{0}_grasp_begin".format(item))
        s += "PandaPose (name {}); ".format("{0}_grasp_end".format(item))
        s += "PandaPose (name {}); ".format("{0}_putdown_begin".format(item))
        s += "PandaPose (name {}); ".format("{0}_putdown_end".format(item))

        item_type = item_types[ind]
        s += "{} (name {}); ".format(item_type, item)
        s += "Target (name {}_init_target); ".format(item)
        s += "Target (name {}_end_target); ".format(item)

    s += "Target (name bin_target); "
    s += "Target (name shelf_target); "
    s += "Target (name off_desk_target); "
    s += "PandaPose (name {}); ".format("robot_init_pose")
    s += "PandaPose (name {}); ".format("robot_end_pose")
    s += "Door (name desk_shelf);"
    s += "Obstacle (name desk_body);"
    s += "Door (name desk_drawer) \n\n"

    s += "Init: "
    rots = {'shelf_handle': [0, 1.57, 0], 'upright_block': [1.57, 0., 0.]}
    for ind, item in enumerate(items):
        dim = dims[ind]
        item_type = item_types[ind]
        if item_type.lower() == 'sphere':
            tail_str = ' {}), '.format(dim[0])
        elif item_type.lower() == 'can':
            tail_str = ' {} {}), '.format(dim[0], dim[1])
        else:
            tail_str = ' {}), '.format(dim)
        rot = rots.get(item, [0.,0.,0.])
        s += "(geom {} {}".format(item, tail_str)
        #s += "(geom {}_init_target {}".format(item, tail_str)
        #s += "(geom {}_end_target {}".format(item, tail_str)

        s += "(pose {0} {1}), ".format(item, init_pos[ind])
        s += "(rotation {0} {1}), ".format(item, rot)

        s += "(value {}_init_target {}), ".format(item, init_pos[ind])
        s += "(rotation {}_init_target {}), ".format(item, rot)

        s += "(value {}_end_target {}), ".format(item, init_pos[ind])
        s += "(rotation {}_end_target {}), ".format(item, rot)

        s += get_undefined_robot_pose_str("{0}_grasp_begin".format(item))
        s += get_undefined_robot_pose_str("{0}_grasp_end".format(item))
        s += get_undefined_robot_pose_str("{0}_putdown_begin".format(item))
        s += get_undefined_robot_pose_str("{0}_putdown_end".format(item))

    s += get_panda_str('panda', R_ARM_INIT, OPEN_GRIPPER, PANDA_INIT_POSE)
    s += get_panda_pose_str('robot_init_pose', R_ARM_INIT, OPEN_GRIPPER, PANDA_INIT_POSE)
    s += get_undefined_robot_pose_str('robot_end_pose')

    s += "(value shelf_target {}), ".format(SHELF_TARGET_POS)
    s += "(rotation shelf_target {}), ".format(SHELF_TARGET_ROT)
    s += "(value bin_target {}), ".format(BIN_TARGET_POS)
    s += "(rotation bin_target {}), ".format(BIN_TARGET_ROT)
    s += "(value off_desk_target {}), ".format(OFF_DESK_TARGET_POS)
    s += "(rotation off_desk_target {}), ".format(OFF_DESK_TARGET_ROT)

    s += "(geom desk_shelf {}), ".format(SHELF_GEOM)
    s += "(pose desk_shelf {}), ".format(SHELF_POS)
    s += "(hinge desk_shelf {}), ".format([0.])
    s += "(rotation desk_shelf {}), ".format(SHELF_ROT)

    s += "(geom desk_body {}), ".format(DESK_BODY_GEOM)
    s += "(pose desk_body {}), ".format(DESK_BODY_POS)
    s += "(rotation desk_body {}), ".format(DESK_BODY_ROT)

    s += "(geom desk_drawer {}), ".format(DRAWER_GEOM)
    s += "(pose desk_drawer {}), ".format(DRAWER_POS)
    s += "(hinge desk_drawer {}), ".format([0.])
    s += "(rotation desk_drawer {}); ".format(DRAWER_ROT)

    for item in items:
        s += "(At {0} {0}_init_target), ".format(item)
    s += "(RobotAt panda robot_init_pose),"
    s += "(StationaryBase panda), "
    s += "(IsMP panda), "
    s += "(WithinJointLimit panda), "
    s += "(StationaryW desk_body) \n\n"

    s += "Goal: {}\n\n".format(GOAL)

    s += "Invariants: "
    s += "(StationaryBase panda), "
    s += "(StationaryRot drawer_handle), "
    s += "(StationaryRot shelf_handle), "
    s += "(StationaryXZ drawer_handle), "
    s += "(StationaryYZ shelf_handle), "
    s += "(StationaryW desk_body), "
    s += "\n\n"

    with open(filename, "w") as f:
        f.write(s)

if __name__ == "__main__":
    main()
