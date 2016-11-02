from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.openrave_body import OpenRAVEBody
from openravepy import matrixFromAxisAngle, IkParameterization, IkParameterizationType, IkFilterOptions, Environment, Planner, RaveCreatePlanner, RaveCreateTrajectory
from sco.expr import Expr
import math
import numpy as np
pi = np.pi

DEFAULT_DIST = 0.6
NUM_BASE_RESAMPLES = 10
EE_ANGLE_SAMPLE_SIZE = 5

OBJ_RING_SAMPLING_RADIUS = 0.6
NUM_EEREACHABLE_RESAMPLE_ATTEMPTS = 10
#These functions are helper functions that can be used by many robots
def get_random_dir():
    """
        This helper function generates a random 2d unit vectors
    """
    rand_dir = np.random.rand(2) - 0.5
    rand_dir = rand_dir/np.linalg.norm(rand_dir)
    return rand_dir

def get_random_theta():
    """
        This helper function generates a random angle between -pi to pi
    """
    theta =  2*np.pi*np.random.rand(1) - np.pi
    return theta[0]

def smaller_ang(x):
    """
        This helper function takes in an angle in radius, and returns smaller angle
        Ex. 5pi/2 -> pi/2
            8pi/3 -> 2pi/3
    """
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

def get_ee_transform_from_pose(pose, rotation):
    """
        This helper function that returns the correct end effector rotation axis (perpendicular to gripper side)
    """
    ee_trans = OpenRAVEBody.transform_from_obj_pose(pose, rotation)
    #the rotation is to transform the tool frame into the end effector transform
    rot_mat = matrixFromAxisAngle([0, np.pi/2, 0])
    ee_rot_mat = ee_trans[:3, :3].dot(rot_mat[:3, :3])
    ee_trans[:3, :3] = ee_rot_mat
    return ee_trans

def closer_joint_angles(pos,seed):
    """
        This helper function cleans up the dof if any angle is greater than 2 pi
    """
    result = np.array(pos)
    for i in [2,4,6]:
        result[i] = closer_ang(pos[i],seed[i],0)
    return result

def get_ee_from_target(targ_pos, targ_rot):
    """
        This function samples all possible EE Poses around the target

        target_pos: position of target we want to sample ee_pose form
        target_rot: rotation of target we want to sample ee_pose form
        return: list of ee_pose tuple in the format of (ee_pos, ee_rot) around target axis
    """
    possible_ee_poses = []
    ee_pos = targ_pos.copy()
    target_trans = OpenRAVEBody.transform_from_obj_pose(targ_pos, targ_rot)
    # rotate can's local z-axis by the amount of linear spacing between 0 to 2pi
    angle_range = np.linspace(0, np.pi*2, num=EE_ANGLE_SAMPLE_SIZE)
    for rot in angle_range:
        target_trans = OpenRAVEBody.transform_from_obj_pose(targ_pos, targ_rot)
        # rotate new ee_pose around can's rotation axis
        rot_mat = matrixFromAxisAngle([0, 0, rot])
        ee_trans = target_trans.dot(rot_mat)
        ee_rot = OpenRAVEBody.obj_pose_from_transform(ee_trans)[3:]
        possible_ee_poses.append((ee_pos, ee_rot))
    return possible_ee_poses

def closest_arm_pose(arm_poses, cur_arm_pose):
    """
        Given a list of possible arm poses, select the one with the least displacement from current arm pose
    """
    min_change = np.inf
    chosen_arm_pose = None
    for arm_pose in arm_poses:
        change = sum((arm_pose - cur_arm_poses)**2)
        if change < min_change:
            chosen_arm_pose = arm_pose
            min_change = change
    return chosen_arm_pose

def closest_base_poses(base_poses, robot_base):
    """
        Given a list of possible base poses, select the one with the least displacement from current base pose
    """
    val, chosen = np.inf, robot_base
    if len(base_poses) <= 0:
        return chosen
    for base_pose in base_poses:
        diff = base_pose - robot_base
        distance = reduce(lambda x, y: x**2 + y, diff, 0)
        if distance < val:
            chosen = base_pose
            val = distance
    return chosen

def lin_interp_traj(start, end, time_steps):
    """
    This helper function returns a linear trajectory from start pose to end pose
    """
    assert start.shape == end.shape
    if time_steps == 0:
        assert np.allclose(start, end)
        return start.copy()
    rows = start.shape[0]
    traj = np.zeros((rows, time_steps+1))

    for i in range(rows):
        traj_row = np.linspace(start[i], end[i], num=time_steps+1)
        traj[i, :] = traj_row
    return traj

def plot_transform(env, T, s=0.1):
    """
    Helper function mainly used for debugging purpose
    Plots transform T in openrave environment.
    S is the length of the axis markers.
    """
    h = []
    x = T[0:3,0]
    y = T[0:3,1]
    z = T[0:3,2]
    o = T[0:3,3]
    h.append(env.drawlinestrip(points=np.array([o, o+s*x]), linewidth=3.0, colors=np.array([(1,0,0),(1,0,0)])))
    h.append(env.drawlinestrip(points=np.array([o, o+s*y]), linewidth=3.0, colors=np.array(((0,1,0),(0,1,0)))))
    h.append(env.drawlinestrip(points=np.array([o, o+s*z]), linewidth=3.0, colors=np.array(((0,0,1),(0,0,1)))))
    return h

def get_expr_mult(coeff, expr):
    """
        Multiply expresions with coefficients
    """
    new_f = lambda x: coeff*expr.eval(x)
    new_grad = lambda x: coeff*expr.grad(x)
    return Expr(new_f, new_grad)

# Sample base values to face the target
def sample_base(target_pose, base_pose):
    vec = target_pose[:2] - np.zeros((2,))
    vec = vec / np.linalg.norm(vec)
    theta = math.atan2(vec[1], vec[0])
    return theta

# Resampling For IK
def get_ik_transform(pos, rot):
    trans = OpenRAVEBody.transform_from_obj_pose(pos, rot)
    # Openravepy flip the rotation axis by 90 degree, thus we need to change it back
    rot_mat = matrixFromAxisAngle([0, np.pi/2, 0])
    trans_mat = trans[:3, :3].dot(rot_mat[:3, :3])
    trans[:3, :3] = trans_mat
    return trans

def get_ik_from_pose(pos, rot, robot, manip_name):
    trans = get_ik_transform(pos, rot)
    solution = get_ik_solutions(robot, manip_name, trans)
    return solution

def get_ik_solutions(robot, manip_name, trans):
    manip = robot.GetManipulator(manip_name)
    iktype = IkParameterizationType.Transform6D
    solutions = manip.FindIKSolutions(IkParameterization(trans, iktype),IkFilterOptions.CheckEnvCollisions)
    return closest_arm_pose(solutions, robot.GetActiveDOFValues()[manip.GetArmIndices])

# Get RRT Planning Result
def get_rrt_traj(env, robot, active_dof, init_dof, end_dof):
    # assert body in env.GetRobot()
    robot.SetActiveDOFs(active_dof)
    robot.SetActiveDOFValues(init_dof)

    params = Planner.PlannerParameters()
    params.SetRobotActiveJoints(robot)
    params.SetGoalConfig(end_dof) # set goal to all ones
    # forces parabolic planning with 40 iterations
    params.SetExtraParameters("""<_postprocessing planner="parabolicsmoother">
        <_nmaxiterations>40</_nmaxiterations>
    </_postprocessing>""")

    planner=RaveCreatePlanner(env,'birrt')
    planner.InitPlan(robot, params)

    traj = RaveCreateTrajectory(env,'')
    result = planner.PlanPath(traj)
    traj_list = []
    for i in range(traj.GetNumWaypoints()):
        # get the waypoint values, this holds velocites, time stamps, etc
        data=traj.GetWaypoint(i)
        # extract the robot joint values only
        dofvalues = traj.GetConfigurationSpecification().ExtractJointValues(data,robot,robot.GetActiveDOFIndices())
        # raveLogInfo('waypint %d is %s'%(i,np.round(dofvalues, 3)))
        traj_list.append(np.round(dofvalues, 3))
    return traj_list

def get_ompl_rrtconnect_traj(env, robot, active_dof, init_dof, end_dof):
    # assert body in env.GetRobot()
    robot.SetActiveDOFs(active_dof)
    robot.SetActiveDOFValues(init_dof)

    params = Planner.PlannerParameters()
    params.SetRobotActiveJoints(robot)
    params.SetGoalConfig(end_dof) # set goal to all ones
    # forces parabolic planning with 40 iterations
    planner=RaveCreatePlanner(env,'OMPL_RRTConnect')
    simplifier = RaveCreatePlanner(env, 'OMPL_Simplifier')
    planner.InitPlan(robot, params)
    traj = RaveCreateTrajectory(env,'')
    planner.PlanPath(traj)
    simplifier.InitPlan(robot, Planner.PlannerParameters())
    simplifier.PlanPath(traj)
    planningutils.RetimeTrajectory(traj)
    planner.PlanPath(traj)

    traj_list = []
    for i in range(traj.GetNumWaypoints()):
        # get the waypoint values, this holds velocites, time stamps, etc
        data=traj.GetWaypoint(i)
        # extract the robot joint values only
        dofvalues = traj.GetConfigurationSpecification().ExtractJointValues(data,robot,robot.GetActiveDOFIndices())
        # raveLogInfo('waypint %d is %s'%(i,np.round(dofvalues, 3)))
        traj_list.append(np.round(dofvalues, 3))
    return traj_list
