from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.openrave_body import OpenRAVEBody
from openravepy import matrixFromAxisAngle, IkParameterization, IkParameterizationType, IkFilterOptions
import math
import numpy as np

pi = np.pi

DEFAULT_DIST = 0.6
NUM_BASE_RESAMPLES = 10

def get_random_dir():
    rand_dir = np.random.rand(2) - 0.5
    rand_dir = rand_dir/np.linalg.norm(rand_dir)
    return rand_dir

def get_random_theta():
    theta =  2*np.pi*np.random.rand(1) - np.pi
    return theta[0]

def sample_base_pose(target_pose, dist=DEFAULT_DIST):
    rand_dir = get_random_dir()
    bp = rand_dir*dist+target_pose[:2]

    vec = target_pose[:2] - bp
    vec = vec / np.linalg.norm(vec)
    theta = math.atan2(vec[1], vec[0])
    pose = np.array([bp[0], bp[1], theta])
    return pose

def get_ee_transform_from_pose(pose, rotation):
    ee_trans = OpenRAVEBody.transform_from_obj_pose(pose, rotation)
    #the rotation is to transform the tool frame into the end effector transform
    rot_mat = matrixFromAxisAngle([0, np.pi/2, 0])
    ee_rot_mat = ee_trans[:3, :3].dot(rot_mat[:3, :3])
    ee_trans[:3, :3] = ee_rot_mat
    return ee_trans

def set_torso_and_arm_to_ik_soln(robot, torso_arm_pose, t):
    torso_pose = torso_arm_pose[:1]
    arm_pose = torso_arm_pose[1:]
    robot.rArmPose[:, t] = arm_pose[:]
    robot.backHeight[:, t] = torso_pose[:]

def get_torso_and_arm_pose_from_ik_soln(ik_solution):
    if ik_solution is None:
        return None, None
    torso_pose = ik_solution[:1]
    arm_pose = ik_solution[1:]
    return torso_pose, arm_pose

def smaller_ang(x):
    return (x + pi)%(2*pi) - pi

def closer_ang(x,a,dir=0):
    """
    find angle y (==x mod 2*pi) that is close to a
    dir == 0: minimize absolute value of difference
    dir == 1: y > x
    dir == 2: y < x
    """
    if dir == 0:
        return a + smaller_ang(x-a)
    elif dir == 1:
        return a + (x-a)%(2*pi)
    elif dir == -1:
        return a + (x-a)%(2*pi) - 2*pi

def closer_joint_angles(pos,seed):
    result = np.array(pos)
    for i in [2,4,6]:
        result[i] = closer_ang(pos[i],seed[i],0)
    return result

# def closer_arm_pose(pos, seed):
#     assert pos.shape == (7,1)
#     assert seed.shape == (7,1)
#     arm_pose = np.zeros((7,1))
#
#     for i in [2,4,6]:
#         arm_pose[i] = closer_ang(pos[i],seed[i],0)
#     return arm_pose

def get_torso_arm_ik(robot_body, target_trans, old_arm_pose=None):
    manip = robot_body.env_body.GetManipulator('rightarm_torso')
    iktype = IkParameterizationType.Transform6D

    solution = manip.FindIKSolution(IkParameterization(target_trans, iktype),IkFilterOptions.CheckEnvCollisions)
    if solution is None:
        return None, None
    torso_pose, arm_pose = get_torso_and_arm_pose_from_ik_soln(solution)
    if old_arm_pose is not None:
        arm_pose = closer_joint_angles(arm_pose, old_arm_pose)
    return torso_pose, arm_pose

def get_col_free_base_pose_around_target(t, plan, target_pose, robot, callback=None, save=False, dist=DEFAULT_DIST):
    base_pose = None
    old_base_pose = robot.pose[:, t].copy()
    for i in range(NUM_BASE_RESAMPLES):
        base_pose = sample_base_pose(target_pose, dist=dist)
        robot.pose[:, t] = base_pose
        if callback is not None: callback()
        _, collision_preds = plan.get_param('RCollides', 1, negated=True, return_preds=True)
        # check to ensure collision_preds are correct

        collision_free = True
        for pred in collision_preds:
            if not pred.test(t, negated=True):
                collision_free = False
                base_pose = None
                break
        if collision_free:
            break

    if not save:
        robot.pose[:, t] = old_base_pose
    return base_pose

def get_col_free_torso_arm_pose(t, pos, rot, robot_param, robot_body,
                                arm_pose_seed=None, save=False, callback=None):
    target_trans = get_ee_transform_from_pose(pos, rot)

    # save arm pose and back height
    old_arm_pose = robot_param.rArmPose[:,t].copy()
    old_back_height = robot_param.backHeight[:,t].copy()

    if arm_pose_seed is None:
        arm_pose_seed = old_arm_pose

    torso_pose, arm_pose = get_torso_arm_ik(robot_body, target_trans,
                                            old_arm_pose=arm_pose_seed)
    if torso_pose is not None:
        robot_param.rArmPose[:, t] = arm_pose
        robot_param.backHeight[:, t] = torso_pose
        if callback is not None:
            trans = OpenRAVEBody.transform_from_obj_pose(pos, rot)
            callback(trans)
            # callback(target_trans)

    # setting parameter values back
    robot_param.rArmPose[:,t] = old_arm_pose
    robot_param.backHeight[:,t] = old_back_height
    return torso_pose, arm_pose
