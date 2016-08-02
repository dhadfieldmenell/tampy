from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.openrave_body import OpenRAVEBody
from openravepy import matrixFromAxisAngle, IkParameterization, IkParameterizationType, IkFilterOptions
import math
import numpy as np

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
    bp = rand_dir[:, None] * dist + target_pose[:2, :]

    vec = target_pose[:2, :] - bp
    vec = vec / np.linalg.norm(vec)
    theta = math.atan2(vec[1], vec[0])
    pose = np.vstack((bp, np.array([[theta]])))
    return pose

def get_ee_transform_from_pose(pose, rotation):
    ee_trans = OpenRAVEBody.transform_from_obj_pose(pose, rotation)
    # Openravepy flip the rotation axis by 90 degree, thus we need to change it back
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
    return torso_pose.reshape((1,1)), arm_pose.reshape((7, 1))

def get_col_free_base_pose_around_target(t, plan, target_pose, robot, callback=None, save=False, dist=DEFAULT_DIST):
    base_pose = None
    old_base_pose = robot.pose[:, t:t+1].copy()
    for i in range(NUM_BASE_RESAMPLES):
        base_pose = sample_base_pose(target_pose, dist=dist)
        robot.pose[:, t:t+1] = base_pose
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
        robot.pose[:, t:t+1] = old_base_pose
    return base_pose

def get_col_free_torso_arm_pose(t, pos, rot, robot_param, robot_body, save=False, callback=None):
    iktype = IkParameterizationType.Transform6D
    manip = robot_body.env_body.GetManipulator('rightarm_torso')
    solution = None
    target_trans = get_ee_transform_from_pose(pos, rot)

    # save arm pose and back height
    old_robot_arm_pose = robot_param.rArmPose[:, t].copy()
    old_robot_back_height = robot_param.backHeight[:, t].copy()

    solution = manip.FindIKSolution(IkParameterization(target_trans, iktype),IkFilterOptions.CheckEnvCollisions)
    if solution is not None:
        set_torso_and_arm_to_ik_soln(robot_param, solution, t)
        if callback is not None: callback(target_trans)
        import ipdb; ipdb.set_trace()
    torso_pose, arm_pose = get_torso_and_arm_pose_from_ik_soln(solution)

    # setting parameter values back
    robot_param.rArmPose[:, t] = old_robot_arm_pose
    robot_param.backHeight[:, t] = old_robot_back_height
    return torso_pose, arm_pose
