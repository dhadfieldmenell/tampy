import copy
import main
import random
import time

from serial import SerialException

import numpy as np

import rospy
from sensor_msgs.msg import Range
from std_msgs.msg import UInt8

import baxter_interface

from core.internal_repr.plan import Plan
from core.parsing import parse_domain_config, parse_problem_config
import core.util_classes.baxter_constants as const
from core.util_classes.viewer import OpenRAVEViewer
from pma.hl_solver import FFSolver
from pma.robot_ll_solver import RobotLLSolver
from ros_interface.basket.basket_predict import BasketPredict
from ros_interface.cloth.cloth_grid_predict import ClothGridPredict
from ros_interface.folding.folding_predict import FoldingPredict
from ros_interface.controllers import EEController, TrajectoryController
from ros_interface.rotate_control import RotateControl
import ros_interface.utils as utils


DOMAIN_FILE = "../domains/laundry_domain/domain.pddl"

RUNNING = 0
PAUSED = 1
SHUTDOWN = 2
ERROR = 3

class HLLaundryState(object):
    # TODO: This class is here for temporary usage but needs generalization
    #       Ideally integreate with existing HLState class and Env Monitor
    def __init__(self):
        self.region_poses = [[], [], [], [], [], [], [], []]
        self.basket_pose = np.array([utils.basket_far_pos[0], utils.basket_far_pos[1], utils.basket_far_rot[0]])
        self.washer_cloth_poses = []

        self.robot_region = 3
        self.washer_door = 0 # -np.pi/2

        self.left_hand_range = 65

        # For constructing high level plans
        self.prob_domain = "(:domain laundry_domain)\n"
        self.objects = "(:objects cloth washer basket)\n"
        self.goal = "(:goal (and (ClothInWasher cloth washer) (not (WasherDoorOpen washer)) (not (ClothInRegion1 cloth)) (not (ClothInRegion2 cloth)) (not (clothInRegion3 Cloth)) (not (ClothInRegion4 cloth)) (not (ClothInBasket cloth basket)) ))\n"

        self.hl_preds = {
            "ClothInBasket": False,
            "ClothInWasher": False,
            "ClothInRegion1": False,
            "ClothInRegion2": False,
            "ClothInRegion3": False,
            "ClothInRegion4": False,
            "BasketInNearLoc": False,
            "BasketInFarLoc": False,
            "BasketNearLocClear": True,
            "BasketFarLocClear": True,
            "WasherDoorOpen": False,
        }

    def get_abs_prob(self):
        prob_str = "(define (problem laundry_prob)\n"
        prob_str += self.prob_domain
        prob_str += self.objects
        prob_str += self.get_init_state()
        prob_str += self.goal
        prob_str += ")\n"
        return prob_str

    def get_init_state(self):
        state_str = "(:init\n"

        if self.hl_preds["ClothInBasket"]:
            state_str += "(ClothInBasket cloth basket)\n"

        if self.hl_preds["ClothInWasher"]:
            state_str += "(ClothInWasher cloth washer)\n"

        if self.hl_preds["ClothInRegion1"]:
            state_str += "(ClothInRegion1 cloth)\n"

        if self.hl_preds["ClothInRegion2"]:
            state_str += "(ClothInRegion2 cloth)\n"

        if self.hl_preds["ClothInRegion3"]:
            state_str += "(ClothInRegion3 cloth)\n"

        if self.hl_preds["ClothInRegion4"]:
            state_str += "(ClothInRegion4 cloth)\n"

        if self.hl_preds["BasketInNearLoc"]:
            state_str += "(BasketInNearLoc basket)\n"

        if self.hl_preds["BasketInFarLoc"]:
            state_str += "(BasketInFarLoc basket)\n"

        if self.hl_preds["BasketNearLocClear"]:
            state_str += "(BasketNearLocClear basket cloth)\n"

        if self.hl_preds["BasketFarLocClear"]:
            state_str += "(BasketFarLocClear basket cloth)\n"

        if self.hl_preds["WasherDoorOpen"]:
            state_str += "(WasherDoorOpen washer)\n"

        state_str += ")\n"
        return state_str

    def store_cloth_regions(self, predictions):
        self.region_poses = [[], [], [], [], [], [], [], []]
        for loc in predictions:
            self.region_poses[loc[1]-1].append(loc[0])
        self.update_cloth_regions()

    def update_cloth_regions(self):
        cloth_in_washer = True
        for i in range(4):
            if len(self.region_poses[i]):
                self.hl_preds["ClothInRegion{0}".format(i+1)] = True
                cloth_in_washer = False
        
        if len(self.region_poses[4]):
            cloth_in_washer = False
            if self.hl_preds["BasketInNearLoc"]:
                self.hl_preds["ClothInBasket"] = True
            else:
                self.hl_preds["BasketNearLocClear"] = False
        elif len(self.region_poses[5]):
            cloth_in_washer = False
            if not self.hl_preds["BasketInNearLoc"]:
                self.hl_preds["BasketNearLocClear"] = False
        else:
            self.hl_preds["BasketNearLocClear"] = True

        if len(self.region_poses[6]):
            cloth_in_washer = False
            if self.hl_preds["BasketInFarLoc"]:
                self.hl_preds["ClothInBasket"] = True
            else:
                self.hl_preds["BasketFarLocClear"] = False
        else:
            self.hl_preds["BasketFarLocClear"] = True

        if ((not self.hl_preds["BasketInNearLoc"]) and len(self.region_poses[4])) and ((not self.hl_preds["BasketInFarLoc"]) and len(self.region_poses[6])):
            self.hl_preds["ClothInBasket"] = False

        self.hl_preds["ClothInWasher"] = cloth_in_washer


    def store_washer_poses(self, poses):
        self.washer_cloth_poses = copy.deepcopy(poses)
        self.update_cloth_in_washer()

    def update_cloth_in_washer(self):
        self.hl_preds["ClothInWasher"] = len(self.washer_cloth_poses) > 0

    def store_basket_loc(self, loc):
        self.basket_pose = loc.copy()
        self.update_basket_loc()

    def update_basket_loc(self):
        self.hl_preds["BasketInNearLoc"] = False
        self.hl_preds["BasketInFarLoc"] = False
        if np.all(np.abs(self.basket_pose[:2] - utils.basket_near_pos[:2]) < 0.15) and np.abs(self.basket_pose[2] - utils.basket_near_rot[0]) < 0.5:
            self.hl_preds["BasketInNearLoc"] = True
            self.hl_preds["ClothInBasket"] = len(self.region_poses[4]) > 0
        elif np.all(np.abs(self.basket_pose[:2] - utils.basket_far_pos[:2]) < 0.15) and np.abs(self.basket_pose[2] - utils.basket_far_rot[0]) < 0.5:
            self.hl_preds["BasketInFarLoc"] = True
            self.hl_preds["ClothInBasket"] = len(self.region_poses[5]) > 0

    def update_door(self, opened):
        self.hl_preds["WasherDoorOpen"] = opened

    def update(self, pred_map):
        for pred, value in pred_map:
            self.hl_preds[pred] = value


class LaundryEnvironmentMonitor(object):
    def __init__(self):
        self.state = HLLaundryState()

        with open(DOMAIN_FILE, 'r+') as f:
            self.abs_domain = f.read()
        self.hl_solver = FFSolver(abs_domain=self.abs_domain)
        self.ll_solver = RobotLLSolver()
        self.env = None
        self.openrave_bodies = {}
        self.has_saved_free_attrs = False
        self.cloth_predictor = ClothGridPredict()
        self.basket_predictor = BasketPredict()
        self.corner_predictor = FoldingPredict()
        self.traj_control = TrajectoryController()
        self.rotate_control = RotateControl()
        self.left_arm = baxter_interface.limb.Limb("left")
        self.left_grip = baxter_interface.gripper.Gripper("left")
        self.left_grip.calibrate()
        self.right_arm = baxter_interface.limb.Limb("right")
        self.right_grip = baxter_interface.gripper.Gripper("right")
        self.right_grip.calibrate()
        self.l_camera = baxter_interface.camera.CameraController("left_hand_camera")
        self.l_camera.open()
        self.l_camera.resolution = (320, 200)
        self.r_camera = baxter_interface.camera.CameraController("right_hand_camera")
        self.r_camera.open()
        self.r_camera.resolution = (320, 200)
        self.range_subscriber = rospy.Subscriber("/robot/range/left_hand_range/state", Range, self._range_callback, queue_size=1)
        self.execution_state = RUNNING
        self.running = True
        self.exec_state_subscriber = rospy.Subscriber("/execution_state", UInt8, self._exec_state_callback, queue_size=1)

    def _range_callback(self, msg):
        self.state.left_hand_range = msg.range

    def _exec_state_callback(self, msg):
        self.execution_state = int(msg.data)

    def run_baxter(self, end_time):
        while self.running:
            success = False
            while not success:
                success = True
                self.predict_basket_location()
                self.predict_cloth_locations()
                hl_plan = self.solve_hl_prob()
                if hl_plan == "Impossible":
                    print "Impossible Plan"
                    import ipdb; ipdb.set_trace()
                    return

                print hl_plan

                for action in hl_plan:
                    while self.execution_state == PAUSED:
                        self.pause()

                    if self.execution_state == SHUTDOWN:
                        self.shutdown()

                    failed = []
                    act_type = action.split()[1].lower()
                    if act_type == "load_basket_from_region_2":
                        failed = self.load_basket_from_region_2()
                    elif act_type == "load_basket_from_region_3":
                        failed = self.load_basket_from_region_3()
                    elif act_type == "load_basket_from_region_4":
                        failed = self.load_basket_from_region_4()
                    elif act_type == "load_washer_from_basket":
                        failed = self.load_washer_from_basket()
                    elif act_type == "load_washer_from_region_1":
                        failed = self.load_washer_from_region_1()
                    elif act_type == "open_washer":
                        failed = self.open_washer()
                    elif act_type == "close_washer":
                        failed = self.close_washer()
                    elif act_type == "move_basket_to_washer":
                        failed = self.move_basket_to_washer()
                    elif act_type == "move_basket_from_washer":
                        failed = self.move_basket_from_washer()
                    elif act_type == "unload_washer_into_basket":
                        failed = self.unload_washer_into_basket()
                    elif act_type == "clear_basket_near_loc":
                        failed = self.clear_basket_near_loc()
                    elif act_type == "clear_basket_far_loc":
                        failed = self.clear_basket_far_loc()
                    else:
                        failed = []

                    if len(failed):
                        self.state.update(failed)
                        success = False
                        break

            self.reset_laundry()

            if time.time() > end_time:
                self.execution_state = SHUTDOWN
                self.shutdown()

    def shutdown(self):
        act_num = 0
        ll_plan_str = []
        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE ROBOT_REGION_1_POSE_0 \n'.format(act_num, self.state.robot_region))
        act_num += 1
        ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_1_POSE_0 ROBOT_REGION_2_POSE_0 REGION2 \n'.format(act_num, self.state.robot_region))
        act_num += 1
        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_REGION_2_POSE_0 END_POSE \n'.format(act_num, self.state.robot_region))

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        success = self.ll_solver._backtrack_solve(plan, callback=None)
        if success:
            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)
        else:
            self.left_arm.move_to_joint_positions({'left_s0':0, 'left_s1':-0.75, 'left_e0':0, 'left_e1':0, 'left_w0':0, 'left_w1':0, 'left_w2':0})
            self.right_arm.move_to_joint_positions({'right_s0':0, 'right_s1':-0.75, 'right_e0':0, 'right_e1':0, 'right_w0':0, 'right_w1':0, 'right_w2':0})


        rs = baxter_interface.robot_enable.RobotDisable()

    def pause(self):
        time.sleep(1)

    def solve_hl_prob(self):
        abs_prob = self.state.get_abs_prob()
        return self.hl_solver._run_planner(self.abs_domain, abs_prob)

    def execute_plan(self, plan, active_ts, limbs=['left', 'right']):
        current_ts = active_ts[0]
        success = True
        self.save_env_bodies(plan.params)
        while (current_ts < active_ts[1] and current_ts < plan.horizon):
            cur_action = filter(lambda a: a.active_timesteps[0] == current_ts, plan.actions)[0]
            if cur_action.name == "open_door":
                self.state.washer_door = -np.pi/2
                self.state.update_door(True)
            elif cur_action.name == "close_door":
                self.state.washer_door = 0
                self.state.update_door(False)

            if cur_action.name.startswith("rotate"):
                old_region = self.state.robot_region
                regions = np.array(utils.regions).flatten()
                self.state.robot_region = np.abs(regions - cur_action.params[-1].value[0,0]).argmin() + 1
                if self.state.robot_region != old_region:
                    try:
                        if np.abs(self.state.robot_region - old_region) < 2:
                            self.rotate_control.rotate_to_region(self.state.robot_region, timeout=20)
                        else:
                            self.rotate_control.rotate_to_region(self.state.robot_region, timeout=35)
                    except SerialException:
                        print "I cannot talk with my rotor base; please make sure all cables are connected properly."
                        return False
            else:
                success = self.traj_control.execute_plan(plan, active_ts=cur_action.active_timesteps, limbs=limbs, check_collision=False)
                if not success:
                    break
            current_ts = cur_action.active_timesteps[1]

        return success

    def predict_cloth_locations(self):
        # act_num = 0
        # ll_plan_str = []
        # ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE BASKET_SCAN_POSE_{1} \n'.format(act_num, self.state.robot_region))
        # act_num += 1
        # ll_plan_str.append('{0}: ROTATE BAXTER BASKET_SCAN_POSE_{1} BASKET_SCAN_POSE_2 REGION2 \n'.format(act_num, self.state.robot_region))
        # act_num += 1

        # plan = self.plan_from_str(ll_plan_str)
        # self.update_plan(plan, {})
        # self.ll_solver._backtrack_solve(plan, callback=None)

        # for action in plan.actions:
        #     self.execute_plan(plan, action.active_timesteps)

        locs = self.cloth_predictor.predict()
        self.state.store_cloth_regions(locs)

    def predict_basket_location(self):
        act_num = 0
        ll_plan_str = []
        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE BASKET_SCAN_POSE_{1} \n'.format(act_num, self.state.robot_region))
        act_num += 1
        ll_plan_str.append('{0}: ROTATE BAXTER BASKET_SCAN_POSE_{1} BASKET_SCAN_POSE_3 REGION3 \n'.format(act_num, self.state.robot_region))
        act_num += 1

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        success = self.ll_solver._backtrack_solve(plan, callback=None)

        if success:
            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)
        else:
            act_num = 0
            ll_plan_str = []
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE ARMS_FORWARD_{1} \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ARMS_FORWARD_{1} BASKET_SCAN_POSE_{2} \n'.format(act_num, self.state.robot_region, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER BASKET_SCAN_POSE_{1} BASKET_SCAN_POSE_3 REGION3 \n'.format(act_num, self.state.robot_region))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {})
            success = self.ll_solver._backtrack_solve(plan, callback=None)
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)


        basket_pose = self.basket_predictor.predict()
        self.state.store_basket_loc(basket_pose)

    def predict_open_door(self):
        self.state.washer_door = self.door_predictor.predict()
        if self.state.washer_door < -np.pi/6:
            self.update_door(False)
        else:
            self.update_door(True)

    def predict_basket_location_from_wrist(self, plan):
        current_angles = self.right.joint_angles()
        current_angles = map(lambda j: current_angles[j], utils.right_joints)
        plan.params['baxter'].set_dof({'rArmPose': current_angles})
        ee_pos = plan.params['baxter'].openrave_body.env_body.GetLink('right_gripper').GetTransfromPose()[-3:]
        self.state.store_basket_loc(self.basket_wrist_predictor.predict_basket(ee_pos))

    def predict_cloth_washer_locations(self):
        washer_cloth_poses = self.cloth_predictor.predict_washer()
        self.state.store_washer_poses(washer_cloth_poses)

    def get_next_cloth_pos(self, region):
        poses = copy.deepcopy(self.state.region_poses)
        if self.state.hl_preds['BasketInFarLoc']:
            poses[0].extend(poses[4])
            poses[1].extend(poses[5])

        if self.state.hl_preds['BasketInNearLoc']:
            poses[2].extend(poses[6])

        next_ind = np.random.choice(range(len(poses[region-1])))
        return np.array(poses[region-1][next_ind])

    def update_plan(self, plan, cloth_to_region={}, in_washer=False, reset_to_region=-1, cloth_z=0.615):
        plan.params['basket'].pose[:2, 0] = self.state.basket_pose[:2]
        plan.params['basket'].rotation[0, 0] = self.state.basket_pose[2]

        left_joint_dict = self.left_arm.joint_angles()
        left_joints = map(lambda j: left_joint_dict[j], const.left_joints)
        left_grip = const.GRIPPER_OPEN_VALUE if self.left_grip.position() > 50 else const.GRIPPER_CLOSE_VALUE
        right_joint_dict = self.right_arm.joint_angles()
        right_joints = map(lambda j: right_joint_dict[j], const.right_joints)
        right_grip = const.GRIPPER_OPEN_VALUE if self.right_grip.position() > 50 else const.GRIPPER_CLOSE_VALUE

        DOF_limits = plan.params['baxter'].openrave_body.env_body.GetDOFLimits()
        left_DOF_limits = (DOF_limits[0][2:9], DOF_limits[1][2:9])
        right_DOF_limits = (DOF_limits[0][10:17], DOF_limits[1][10:17])

        left_joints = np.maximum(left_DOF_limits[0], left_joints)
        left_joints = np.minimum(left_DOF_limits[1], left_joints)
        right_joints = np.maximum(right_DOF_limits[0], right_joints)
        right_joints = np.minimum(right_DOF_limits[1], right_joints)
        plan.params['baxter'].lArmPose[:, 0] = left_joints
        plan.params['baxter'].lGripper[:, 0] = left_grip
        plan.params['baxter'].rArmPose[:, 0] = right_joints
        plan.params['baxter'].rGripper[:, 0] = right_grip
        plan.params['baxter'].pose[:,0] = utils.regions[self.state.robot_region-1]

        plan.params['robot_init_pose'].lArmPose[:, 0] = left_joints
        plan.params['robot_init_pose'].lGripper[:, 0] = left_grip
        plan.params['robot_init_pose'].rArmPose[:, 0] = right_joints
        plan.params['robot_init_pose'].rGripper[:, 0] = right_grip
        plan.params['robot_init_pose'].value[:,0] = plan.params['baxter'].pose[:,0]

        plan.params['washer'].door[:,0] = self.state.washer_door

        if self.state.hl_preds['BasketInFarLoc']:
            plan.params['basket_far_target'].value[:,0] = plan.params['basket'].pose[:, 0]
            plan.params['basket_far_target'].rotation[:,0] = plan.params['basket'].rotation[:, 0]

        if self.state.hl_preds['BasketInNearLoc']:
            plan.params['basket_near_target'].value[:,0] = plan.params['basket'].pose[:, 0]
            plan.params['basket_near_target'].rotation[:,0] = plan.params['basket'].rotation[:, 0]

        for cloth in cloth_to_region:
            pose = cloth_to_region[cloth]
            plan.params['{0}'.format(cloth)].pose[:2, 0] = pose + np.random.uniform(0, 0.02, (2,))
            plan.params['{0}'.format(cloth)].pose[2, 0] = cloth_z
            plan.params['cloth_target_begin_{0}'.format(cloth[-1])].value[:, 0] = plan.params['{0}'.format(cloth)].pose[:, 0]

        if len(self.state.washer_cloth_poses) and in_washer:
            plan.params['cloth0'].pose[:, 0] = plan.params['washer'].pose[:,0] + np.array([const.WASHER_DEPTH_OFFSET/2, np.sqrt(3)*const.WASHER_DEPTH_OFFSET/2, -.14]) # self.state.washer_cloth_poses[0]
            plan.params['cloth_target_begin_0'].value[:, 0] = plan.params['cloth0'].pose[:, 0]

        if reset_to_region > 0:
            accept = False
            while not accept:
                filter_poses = lambda p: p[1] == reset_to_region or (reset_to_region == 1 and p[1] == 5)
                possible_poses = filter(filter_poses, utils.cloth_grid_coordinates)
                next_pose = random.choice(possible_poses)
                ref = utils.cloth_grid_ref
                disp = np.array(ref[0] - next_pose[0]) / utils.pixels_per_cm
                true_pose = (ref[1] + disp) / 100.0
                plan.params['baxter'].openrave_body.set_pose([0, 0, utils.regions[reset_to_region-1][0]])
                accept = len(plan.params['baxter'].openrave_body.get_ik_from_pose(np.r_[true_pose, 0.8], [0, np.pi/2, 0], "left_arm")) > 0
                plan.params['cloth_target_end_0'].value[:2,0] = true_pose
                plan.params['cloth_target_end_0'].value[2, :] = cloth_z
                print true_pose

    def save_env_bodies(self, params):
        for param_name in params:
            param = params[param_name]
            if not param.is_symbol() and param.openrave_body is not None:
                self.openrave_bodies[param_name] = param.openrave_body

    def restore_env_bodies(self, params):
        for param_name in params:
            param = params[param_name]
            if param_name in self.openrave_bodies:
                param.openrave_body = self.openrave_bodies[param_name]

    def plan_from_str(self, ll_plan_str):
        '''Convert a plan string (low level) into a plan object.'''
        num_cloth = 1
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = FFSolver(d_c)
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_{0}.prob'.format(num_cloth))
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, self.env, self.openrave_bodies)
        if self.env is None:
            self.env = problem.env
            # self.env.SetViewer('qtcoin')
        plan = hls.get_plan(ll_plan_str, domain, problem)
        self.restore_env_bodies(plan.params)
        return plan

    # def load_basket_from_region_1(self):
    #     self.predict_basket_location()
    #     self.predict_cloth_locations()

    #     failed_constr = []
    #     if not len(self.state.region_poses[0]):
    #          failed_constr.append(("ClothInRegion1", False))

    #     if np.any(np.abs(self.state.basket_pose - utils.basket_far_pos)[:2] > 0.15):
    #         failed_constr.append(("BasketInFarLoc", False))

    #     if len(failed_constr):
    #         return failed_constr

    #     act_num = 0
    #     ll_plan_str = []

    #     if (self.state.robot_region != 1):
    #         ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
    #         act_num += 1
    #         ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
    #         act_num += 1

    #     while len(self.state.region_poses[0]):
    #         ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_1_POSE_0 CLOTH_GRASP_BEGIN_0 \n'.format(act_num))
    #         act_num += 1
    #         ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5} \n'.format(act_num))
    #         act_num += 1
    #         ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 CLOTH_PUTDOWN_BEGIN_{2} CLOTH{3} \n'.format(act_num))
    #         act_num += 1
    #         ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH{1} BASKET CLOTH_TARGET_END_{2} BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5} \n'.format(act_num))
    #         act_num += 1

    #         plan = self.plan_from_str(ll_plan_str)
    #         self.update_plan(plan, {'cloth0': 1})
    #         self.ll_solver.backtrack_solve(plan, callback=None)

    #         for action in plan.actions:
    #             self.execute_plan(plan, action.active_timesteps)

    #         act_num = 0
    #         ll_plan_str = []
    #         self.predict_cloth_locations()
    #         # self.predict_basket_location()

    #         # if np.any(np.abs(self.state.basket_pose - utils.basket_far_pos)[:2] > 0.1):
    #         #     return [("BasketNearWasher", True)]

    #     return []

    def load_basket_in_region_1(self):
        self.predict_cloth_locations()

        failed_constr = []
        if not len(self.state.region_poses[0]):
             failed_constr.append(("ClothInRegion1", False))

        if np.any(np.abs(self.state.basket_pose - utils.basket_near_pos)[:2] > 0.15):
            failed_constr.append(("BasketInNearLoc", False))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        while len(self.state.region_poses[2]):
            act_num = 0
            ll_plan_str = []

            last_pose = 'ROBOT_INIT_POSE'
            if (self.state.robot_region != 1):
                ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                last_pose = 'ROBOT_REGION_1_POSE_0'
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER {1} CLOTH_GRASP_BEGIN_0 \n'.format(act_num, last_pose))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1


            plan = self.plan_from_str(ll_plan_str)
            cloth_pos = self.get_next_cloth_pos(1)
            self.update_plan(plan, {'cloth0': cloth_pos})
            self.ll_solver._backtrack_solve(plan, callback=None, amax=len(plan.actions)-2)

            for action in plan.actions[:-1]:
                self.execute_plan(plan, action.active_timesteps)

            final_offset = np.array([0.0,0.0])
            for offset in utils.wrist_im_offsets:
                cloth_present = self.cloth_predictor.predict_wrist_center(offset[0])
                if cloth_present:
                    plan.params['baxter'].openrave_body.set_pose([0,0,utils.regions[2][0]])
                    accept = len(plan.params['baxter'].openrave_body.get_ik_from_pose(np.r_[cloth_pos[:2], 0.725]+np.r_[offset[1], 0]+np.r_[final_offset, 0], [0, np.pi/2, 0], "left_arm")) > 0
                    if accept:
                        final_offset += np.array(offset[1])

            cloth_pos[:2] += final_offset

            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 ROBOT_INIT_POSE CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 LOAD_WASHER_INTERMEDIATE_POSE CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER LOAD_WASHER_INTERMEDIATE_POSE LOAD_BASKET_NEAR CLOTH0 \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': cloth_pos})
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            dropped_cloth = False
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)
            else:
                continue

            if self.state.left_hand_range > 1:
                self.left_grip.open()
                dropped_cloth = True
                continue


            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_INIT_POSE CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER CLOTH_PUTDOWN_END_0 LOAD_BASKET_NEAR \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan)
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            dropped_cloth = False
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)

            act_num = 0
            ll_plan_str = []
            self.predict_cloth_locations()

        return []



    def load_basket_from_region_2(self):
        self.predict_basket_location()
        self.predict_cloth_locations()

        failed_constr = []
        if not len(self.state.region_poses[1]):
             failed_constr.append(("ClothInRegion2", False))

        if np.any(np.abs(self.state.basket_pose - utils.basket_far_pos)[:2] > 0.15):
            failed_constr.append(("BasketInFarLoc", False))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        while len(self.state.region_poses[1]):
            act_num = 0
            ll_plan_str = []

            start_pose = 'ROBOT_INIT_POSE'
            if (self.state.robot_region != 2):
                ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_2_POSE_0 REGION2 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                start_pose = 'ROBOT_REGION_2_POSE_0'
            ll_plan_str.append('{0}: MOVETO BAXTER {1} START_GRASP_2 \n'.format(act_num, start_pose))
            act_num += 1
            ll_plan_str.append('{0}: MOVETO BAXTER START_GRASP_2 CLOTH_GRASP_BEGIN_0 \n'.format(act_num, start_pose))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1


            plan = self.plan_from_str(ll_plan_str)
            cloth_pos = self.get_next_cloth_pos(2)
            self.update_plan(plan, {'cloth0': cloth_pos})
            self.ll_solver._backtrack_solve(plan, callback=None, amax=len(plan.actions)-2)

            for action in plan.actions[:-1]:
                self.execute_plan(plan, action.active_timesteps)

            final_offset = np.array([0.0,0.0])
            for offset in utils.wrist_im_offsets:
                cloth_present = self.cloth_predictor.predict_wrist_center(offset[0])
                if cloth_present:
                    plan.params['baxter'].openrave_body.set_pose([0,0,utils.regions[1][0]])
                    accept = len(plan.params['baxter'].openrave_body.get_ik_from_pose(np.r_[cloth_pos[:2], 0.725]+np.r_[offset[1], 0]+np.r_[final_offset, 0], [0, np.pi/2, 0], "left_arm")) > 0
                    if accept:
                        final_offset += np.array(offset[1])

            cloth_pos[:2] += final_offset

            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 ROBOT_INIT_POSE CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 LOAD_BASKET_INTER_2 CLOTH0 \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': cloth_pos})
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            dropped_cloth = False
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)
            else:
                self.predict_cloth_locations()
                continue

            if self.state.left_hand_range > 1:
                self.left_grip.open()
                dropped_cloth = True
                self.predict_cloth_locations()
                continue


            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: ROTATE_HOLDING_CLOTH BAXTER CLOTH0 ROBOT_INIT_POSE ROBOT_REGION_3_POSE_1 REGION3 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_REGION_3_POSE_1 CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_FAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': cloth_pos})
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            dropped_cloth = False
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)
                    if action in plan.actions[:1] and self.state.left_hand_range > 1:
                        self.left_grip.open()
                        dropped_cloth = True
                        break

            act_num = 0
            ll_plan_str = []
            self.predict_cloth_locations()
            # self.predict_basket_location()

            # if np.any(np.abs(self.state.basket_pose - utils.basket_far_pos) > 0.05):
            #     failed_constr.append(("BasketNearWasher", True))

        return []

    def load_basket_from_region_3(self):
        self.predict_basket_location()
        self.predict_cloth_locations()

        failed_constr = []
        if not len(self.state.region_poses[2]):
             failed_constr.append(("ClothInRegion3", False))

        if np.any(np.abs(self.state.basket_pose - utils.basket_far_pos)[:2] > 0.15):
            failed_constr.append(("BasketInFarLoc", False))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        while len(self.state.region_poses[2]):
            act_num = 0
            ll_plan_str = []

            last_pose = 'ROBOT_INIT_POSE'
            if (self.state.robot_region != 3):
                ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_3_POSE_0 REGION3 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                last_pose = 'ROBOT_REGION_3_POSE_0'
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER {1} CLOTH_GRASP_BEGIN_0 \n'.format(act_num, last_pose))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1


            plan = self.plan_from_str(ll_plan_str)
            cloth_pos = self.get_next_cloth_pos(3)
            self.update_plan(plan, {'cloth0': cloth_pos})
            self.ll_solver._backtrack_solve(plan, callback=None, amax=len(plan.actions)-2)

            for action in plan.actions[:-1]:
                self.execute_plan(plan, action.active_timesteps)

            final_offset = np.array([0.0,0.0])
            for offset in utils.wrist_im_offsets:
                cloth_present = self.cloth_predictor.predict_wrist_center(offset[0])
                if cloth_present:
                    plan.params['baxter'].openrave_body.set_pose([0,0,utils.regions[2][0]])
                    accept = len(plan.params['baxter'].openrave_body.get_ik_from_pose(np.r_[cloth_pos[:2], 0.725]+np.r_[offset[1], 0]+np.r_[final_offset, 0], [0, np.pi/2, 0], "left_arm")) > 0
                    if accept:
                        final_offset += np.array(offset[1])

            cloth_pos[:2] += final_offset

            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 ROBOT_INIT_POSE CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 LOAD_BASKET_FAR CLOTH0 \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': cloth_pos})
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            dropped_cloth = False
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)
            else:
                continue

            if self.state.left_hand_range > 1:
                self.left_grip.open()
                dropped_cloth = True
                continue


            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_INIT_POSE CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_FAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER CLOTH_PUTDOWN_END_0 LOAD_BASKET_FAR \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan)
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            dropped_cloth = False
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)

            act_num = 0
            ll_plan_str = []
            self.predict_cloth_locations()

        return []

    def load_basket_from_region_4(self):
        self.predict_basket_location()
        self.predict_cloth_locations()

        failed_constr = []
        if not len(self.state.region_poses[3]):
             failed_constr.append(("ClothInRegion4", False))

        if np.any(np.abs(self.state.basket_pose - utils.basket_far_pos)[:2] > 0.15):
            failed_constr.append(("BasketInFarLoc", False))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        while len(self.state.region_poses[1]):
            act_num = 0
            ll_plan_str = []

            start_pose = 'ROBOT_INIT_POSE'
            if (self.state.robot_region != 4):
                ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_4_POSE_0 REGION4 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                start_pose = 'ROBOT_REGION_4_POSE_0'
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER {1} CLOTH_GRASP_BEGIN_0 \n'.format(act_num, start_pose))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1


            plan = self.plan_from_str(ll_plan_str)
            cloth_pos = self.get_next_cloth_pos(4)
            self.update_plan(plan, {'cloth0': cloth_pos})
            success = self.ll_solver._backtrack_solve(plan, callback=None, amax=len(plan.actions)-2)

            for action in plan.actions[:-1]:
                self.execute_plan(plan, action.active_timesteps)

            final_offset = np.array([0.0,0.0])
            for offset in utils.wrist_im_offsets:
                cloth_present = self.cloth_predictor.predict_wrist_center(offset[0])
                if cloth_present:
                    plan.params['baxter'].openrave_body.set_pose([0,0,utils.regions[3][0]])
                    accept = len(plan.params['baxter'].openrave_body.get_ik_from_pose(np.r_[cloth_pos[:2], 0.725]+np.r_[offset[1], 0]+np.r_[final_offset, 0], [0, np.pi/2, 0], "left_arm")) > 0
                    if accept:
                        final_offset += np.array(offset[1])

            cloth_pos[:2] += final_offset

            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 ROBOT_INIT_POSE CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 LOAD_BASKET_INTER_4 CLOTH0 \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': cloth_pos})
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            dropped_cloth = False
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)
            else:
                continue

            if self.state.left_hand_range > 1:
                self.left_grip.open()
                dropped_cloth = True
                continue


            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: ROTATE_HOLDING_CLOTH BAXTER CLOTH0 ROBOT_INIT_POSE ROBOT_REGION_3_POSE_1 REGION3 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_REGION_3_POSE_1 CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_FAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': cloth_pos})
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            dropped_cloth = False
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)
                    if action in plan.actions[:1] and self.state.left_hand_range > 1:
                        self.left_grip.open()
                        dropped_cloth = True
                        break

            act_num = 0
            ll_plan_str = []
            # self.predict_basket_location()
            self.predict_cloth_locations()

            # if np.any(np.abs(self.state.basket_pose - utils.basket_far_pos) > 0.05):
            #     failed_constr.append(("BasketNearWasher", True))

        return []

    def move_basket_to_washer(self):
        self.predict_basket_location()
        self.predict_cloth_locations()

        failed_constr = []
        # if len(self.state.region_poses[4]) or len(self.state.region_poses[5]):
        #      failed_constr.append(("BasketNearLocClear", False))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        last_pose = 'ROBOT_INIT_POSE'
        if (self.state.robot_region != 3):
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_3_POSE_0 REGION3 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            last_pose = 'ROBOT_REGION_3_POSE_0'

        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER {1} BASKET_GRASP_BEGIN_0 \n'.format(act_num, last_pose))
        act_num += 1
        ll_plan_str.append('{0}: BASKET_GRASP BAXTER BASKET BASKET_FAR_TARGET BASKET_GRASP_BEGIN_0 BG_EE_LEFT_0 BG_EE_RIGHT_0 BASKET_GRASP_END_0 \n'.format(act_num))
        act_num += 1

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        success = self.ll_solver._backtrack_solve(plan, callback=None, amax=len(plan.actions)-2)

        if success:
            for action in plan.actions[:-1]:
                self.execute_plan(plan, action.active_timesteps)

        # self.predict_basket_location_from_wrist(plan)
        cloth_on_handle = self.cloth_predictor.predict_wrist_center()
        if cloth_on_handle:                
            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP_FROM_HANDLE BAXTER CLOTH0 BASKET BASKET_FAR_TARGET CLOTH_GRASP_BEGIN_0 BG_EE_LEFT_0 BG_EE_RIGHT_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 LOAD_BASKET_FAR CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER LOAD_BASKET_FAR CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_FAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            
            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {})
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps, limbs=['left'])

            self.predict_basket_location()

        act_num = 0
        ll_plan_str = []

        last_pose = 'ROBOT_INIT_POSE'
        if (self.state.robot_region != 3):
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_3_POSE_0 REGION3 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            last_pose = 'ROBOT_REGION_3_POSE_0'

        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER {1} BASKET_GRASP_BEGIN_0 \n'.format(act_num, last_pose))
        act_num += 1
        ll_plan_str.append('{0}: BASKET_GRASP BAXTER BASKET BASKET_FAR_TARGET BASKET_GRASP_BEGIN_0 BG_EE_LEFT_0 BG_EE_RIGHT_0 BASKET_GRASP_END_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: ROTATE_HOLDING_BASKET BAXTER BASKET BASKET_GRASP_END_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVEHOLDING_BASKET BAXTER ROBOT_REGION_1_POSE_0 BASKET_PUTDOWN_BEGIN_0 BASKET \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BASKET_PUTDOWN BAXTER BASKET BASKET_NEAR_TARGET BASKET_PUTDOWN_BEGIN_0 BP_EE_LEFT_0 BP_EE_RIGHT_0 BASKET_PUTDOWN_END_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER BASKET_PUTDOWN_END_0 ARMS_FORWARD_1 \n'.format(act_num))

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        success = self.ll_solver.backtrack_solve(plan, callback=None)

        if success:
            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

            if action.name == 'basket_grasp':
                success = success or self.traj_control.left_grip.position() < 20 or self.traj_control.right_grip.position() < 20

            if not success:
                self.traj_control.left_grip.open()
                self.traj_control.right_grip.open()
                return [('BasketInFarLoc', False)]

        act_num = 0
        ll_plan_str = []

        return []

    def move_basket_from_washer(self):
        self.predict_basket_location()
        self.predict_cloth_locations()

        failed_constr = []
        if len(self.state.region_poses[6]):
             failed_constr.append(("BasketFarLocClear", False))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        last_pose = 'ROBOT_INIT_POSE'
        if (self.state.robot_region != 1):
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            last_pose = 'ROBOT_REGION_1_POSE_0'

        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER {1} BASKET_GRASP_BEGIN_0 \n'.format(act_num, last_pose))
        act_num += 1
        ll_plan_str.append('{0}: BASKET_GRASP BAXTER BASKET BASKET_NEAR_TARGET BASKET_GRASP_BEGIN_0 BG_EE_LEFT_0 BG_EE_RIGHT_0 BASKET_GRASP_END_0 \n'.format(act_num))
        act_num += 1

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        success = self.ll_solver._backtrack_solve(plan, callback=None, amax=len(plan.actions)-2)

        if success:
            for action in plan.actions[:-1]:
                self.execute_plan(plan, action.active_timesteps)

        # cloth_on_handle = self.cloth_predictor.predict_wrist_center()
        # if cloth_on_handle:                
        #     act_num = 0
        #     ll_plan_str = []

        #     ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0 \n'.format(act_num))
        #     act_num += 1
        #     ll_plan_str.append('{0}: CLOTH_GRASP_FROM_HANDLE BAXTER CLOTH0 BASKET BASKET_NEAR_TARGET CLOTH_GRASP_BEGIN_0 BG_EE_LEFT_0 BG_EE_RIGHT_0 CLOTH_GRASP_END_0 \n'.format(act_num))
        #     act_num += 1
        #     ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 LOAD_BASKET_NEAR CLOTH0 \n'.format(act_num))
        #     act_num += 1
        #     ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER LOAD_BASKET_NEAR CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
        #     act_num += 1
        #     ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            
        #     plan = self.plan_from_str(ll_plan_str)
        #     self.update_plan(plan, {})
        #     success = self.ll_solver.backtrack_solve(plan, callback=None)

        #     if success:
        #         for action in plan.actions:
        #             self.execute_plan(plan, action.active_timesteps, limbs=['left'])

        #     self.predict_basket_location()

        act_num = 0
        ll_plan_str = []

        last_pose = 'ROBOT_INIT_POSE'
        if (self.state.robot_region != 1):
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            last_pose = 'ROBOT_REGION_1_POSE_0'

        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER {1} BASKET_GRASP_BEGIN_0 \n'.format(act_num, last_pose))
        act_num += 1
        ll_plan_str.append('{0}: BASKET_GRASP BAXTER BASKET BASKET_NEAR_TARGET BASKET_GRASP_BEGIN_0 BG_EE_LEFT_0 BG_EE_RIGHT_0 BASKET_GRASP_END_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: ROTATE_HOLDING_BASKET BAXTER BASKET BASKET_GRASP_END_0 ROBOT_REGION_3_POSE_1 REGION3 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVEHOLDING_BASKET BAXTER ROBOT_REGION_3_POSE_1 BASKET_PUTDOWN_BEGIN_0 BASKET \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BASKET_PUTDOWN BAXTER BASKET BASKET_FAR_TARGET BASKET_PUTDOWN_BEGIN_0 BP_EE_LEFT_0 BP_EE_RIGHT_0 BASKET_PUTDOWN_END_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER BASKET_PUTDOWN_END_0 ARMS_FORWARD_3 \n'.format(act_num))

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        success = self.ll_solver.backtrack_solve(plan, callback=None)

        if success:
            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

            if action.name == 'basket_grasp':
                success = success or self.traj_control.left_grip.position() < 20 or self.traj_control.right_grip.position() < 20
                
            if not success:
                self.traj_control.left_grip.open()
                self.traj_control.right_grip.open()
                return [('BasketInFarLoc', False)]

        act_num = 0
        ll_plan_str = []

        return []


    def open_washer(self):
        failed_constr = []

        act_num = 0
        ll_plan_str = []

        last_pose = 'ROBOT_INIT_POSE'
        if (self.state.robot_region != 1):
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            last_pose = 'ROBOT_REGION_1_POSE_0'

        # ll_plan_str.append('{0}: MOVETO BAXTER {1} CLOSE_DOOR_SCAN_POSE \n'.format(act_num, last_pose))

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {})
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)

            # self.predict_open_door()
            if self.state.washer_door < -np.pi/6:
                 failed_constr.append(("WasherDoorOpen", True))

            if len(failed_constr):
                return failed_constr
        
        act_num = 0
        ll_plan_str = []

        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE BASKET_SCAN_POSE_1 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER BASKET_SCAN_POSE_1 OPEN_DOOR_BEGIN_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: OPEN_DOOR BAXTER WASHER OPEN_DOOR_BEGIN_0 OPEN_DOOR_EE_APPROACH_0 OPEN_DOOR_EE_RETREAT_0 OPEN_DOOR_END_0 WASHER_CLOSE_POSE_0 WASHER_OPEN_POSE_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER OPEN_DOOR_END_0 ARM_BACK_1 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ARM_BACK_1 ROBOT_INIT_POSE \n'.format(act_num))
        act_num += 1
        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        success = self.ll_solver.backtrack_solve(plan, callback=None)

        if success:
            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

        self.state.washer_door = -np.pi/2

        return []

    def close_washer(self):
        # self.predict_basket_location()
        self.predict_cloth_locations()
        failed_constr = []
        if len(self.state.region_poses[0]):
            failed_constr.append(("ClothInRegion1", True))

        # if len(self.state.region_poses[1]):
        #     failed_constr.append(("ClothInRegion2", True))

        # if len(self.state.region_poses[2]):
        #     failed_constr.append(("ClothInRegion3", True))

        # if len(self.state.region_poses[3]):
        #     failed_constr.append(("ClothInRegion4", True))

        if self.state.washer_door > -np.pi/6:
             failed_constr.append(("WasherDoorOpen", False))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        last_pose = 'ROBOT_INIT_POSE'
        if (self.state.robot_region != 1):
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            last_pose = 'ROBOT_REGION_1_POSE_0'

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {})
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)

            if len(failed_constr):
                return failed_constr

        self.state.hl_preds['WasherDoorOpen'] = False
        self.state.washer_door = 0

        act_num = 0
        ll_plan_str = []

        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER {1} BASKET_SCAN_POSE_1 \n'.format(act_num, last_pose))
        act_num += 1
        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER BASKET_SCAN_POSE_1 PUSH_WASHER_CLOSE_BEGIN \n'.format(act_num, last_pose))
        act_num += 1
        # ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ARM_BACK_1 CLOSE_DOOR_BEGIN_0 \n'.format(act_num))
        # act_num += 1
        # ll_plan_str.append('{0}: CLOSE_DOOR BAXTER WASHER CLOSE_DOOR_BEGIN_0 CLOSE_DOOR_EE_APPROACH_0 CLOSE_DOOR_EE_RETREAT_0 CLOSE_DOOR_END_0 WASHER_OPEN_POSE_0 WASHER_CLOSE_POSE_0 \n'.format(act_num))
        # act_num += 1
        # ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER CLOSE_DOOR_END_0 PUSH_WASHER_CLOSE_BEGIN \n'.format(act_num))
        # act_num += 1
        ll_plan_str.append('{0}: MOVE_NO_COLLISION_CHECK BAXTER PUSH_WASHER_CLOSE_BEGIN PUSH_WASHER_CLOSE_END \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVE_NO_COLLISION_CHECK BAXTER PUSH_WASHER_CLOSE_END PUSH_WASHER_CLOSE_BEGIN \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER PUSH_WASHER_CLOSE_BEGIN BASKET_SCAN_POSE_1 \n'.format(act_num))
        act_num += 1
        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        success = self.ll_solver.backtrack_solve(plan, callback=None)

        if success:
            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

        self.state.washer_door = 0

        return []

    def load_washer_from_basket(self):
        self.predict_basket_location()
        self.predict_cloth_locations()

        failed_constr = []
        if not len(self.state.region_poses[4]):
             failed_constr.append(("ClothInBasket", False))

        if self.state.washer_door > -np.pi/3:
            failed_constr.append(("WasherDoorOpen", False))

        if np.any(np.abs(self.state.basket_pose[:2] - utils.basket_near_pos[:2]) > 0.1):
            failed_constr.append(("BasketInNearLoc", False))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        while len(self.state.region_poses[4]):
            act_num = 0
            ll_plan_str = []

            last_pose = 'ROBOT_INIT_POSE'
            if (self.state.robot_region != 1):
                ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                last_pose = 'ROBOT_REGION_1_POSE_0'

            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER {1} LOAD_BASKET_NEAR \n'.format(act_num, last_pose))
            act_num += 1
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER LOAD_BASKET_NEAR CLOTH_GRASP_BEGIN_0 \n'.format(act_num, last_pose))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))


            plan = self.plan_from_str(ll_plan_str)
            cloth_pos = self.get_next_cloth_pos(5)
            self.update_plan(plan, {'cloth0': cloth_pos})
            self.ll_solver._backtrack_solve(plan, callback=None, amax=len(plan.actions)-2)

            for action in plan.actions[:-1]:
                self.execute_plan(plan, action.active_timesteps)

            final_offset = np.array([0.0,0.0])
            for offset in utils.wrist_im_offsets:
                cloth_present = self.cloth_predictor.predict_wrist_center(offset[0])
                if cloth_present:
                    plan.params['baxter'].openrave_body.set_pose([0,0,utils.regions[0][0]])
                    accept = len(plan.params['baxter'].openrave_body.get_ik_from_pose(np.r_[cloth_pos[:2], 0.725]+np.r_[offset[1], 0]+np.r_[final_offset, 0], [0, np.pi/2, 0], "left_arm")) > 0
                    if accept:
                        final_offset += np.array(offset[1])

            cloth_pos[:2] += final_offset

            act_num = 0
            ll_plan_str = []

            cloth_range = self.state.left_hand_range
            cloth_z = 0.635
            # if cloth_range < 0.15:
            #     cloth_z = 0.7
            # elif cloth_range < 0.25:
            #     cloth_z = 0.67

            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 ROBOT_INIT_POSE CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 LOAD_BASKET_NEAR CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER LOAD_BASKET_NEAR LOAD_WASHER_INTERMEDIATE_POSE CLOTH0 \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': cloth_pos}, cloth_z=cloth_z)
            success = self.ll_solver.backtrack_solve(plan, callback=None)
            
            act_num = 0
            ll_plan_str = []

            dropped_cloth = False
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)
            else:
                continue
            
            if self.state.left_hand_range > 1:
                self.left_grip.open()
                dropped_cloth = True

            while not dropped_cloth:
                ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_INIT_POSE PUT_INTO_WASHER_BEGIN CLOTH0 \n'.format(act_num))
                act_num += 1
                ll_plan_str.append('{0}: PUT_INTO_WASHER BAXTER WASHER WASHER_OPEN_POSE_0 CLOTH0 CLOTH_TARGET_END_0 PUT_INTO_WASHER_BEGIN CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
                act_num += 1
                ll_plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_0 INTERMEDIATE_UNLOAD \n'.format(act_num))
                act_num += 1
                ll_plan_str.append('{0}: MOVETO BAXTER INTERMEDIATE_UNLOAD IN_WASHER_ADJUST \n'.format(act_num))
                act_num += 1
                ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER IN_WASHER_ADJUST IN_WASHER_ADJUST_2 CLOTH0 \n'.format(act_num))
                act_num += 1
                ll_plan_str.append('{0}: MOVETO BAXTER IN_WASHER_ADJUST_2 INTERMEDIATE_UNLOAD \n'.format(act_num))
                act_num += 1
                ll_plan_str.append('{0}: MOVETO BAXTER INTERMEDIATE_UNLOAD LOAD_WASHER_INTERMEDIATE_POSE \n'.format(act_num))
                act_num += 1

                plan = self.plan_from_str(ll_plan_str)
                self.update_plan(plan, {'cloth0': cloth_pos}, cloth_z=cloth_z)
                success = self.ll_solver.backtrack_solve(plan, callback=None)

                dropped_cloth = True
                if success:
                    for action in plan.actions:
                        self.execute_plan(plan, action.active_timesteps)

                if self.state.left_hand_range < 1:
                    self.left_grip.close()
                    dropped_cloth = False

                act_num = 0
                ll_plan_str = []
            self.predict_cloth_locations()

        act_num = 0
        ll_plan_str = []
        ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE LOAD_BASKET_NEAR \n'.format(act_num))

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan)
        success = self.ll_solver.backtrack_solve(plan, callback=None)

        if success:
            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

        return []

    def load_washer_from_region_1(self):
        self.predict_basket_location()
        self.predict_cloth_locations()

        failed_constr = []
        if not len(self.state.region_poses[0]):
             failed_constr.append(("ClothInRegion1", False))

        if self.state.washer_door > -np.pi/2:
            failed_constr.append(("WasherDoorOpen", False))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        while len(self.state.region_poses[0]):
            act_num = 0
            ll_plan_str = []

            last_pose = 'ROBOT_INIT_POSE'
            if (self.state.robot_region != 1):
                ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                last_pose = 'ROBOT_REGION_1_POSE_0'

            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER {1} LOAD_BASKET_NEAR \n'.format(act_num, last_pose))
            act_num += 1
            ll_plan_str.append('{0}: MOVETO BAXTER LOAD_BASKET_NEAR CLOTH_GRASP_BEGIN_0 \n'.format(act_num, last_pose))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))


            plan = self.plan_from_str(ll_plan_str)
            cloth_pos = self.get_next_cloth_pos(1)
            self.update_plan(plan, {'cloth0': cloth_pos})
            self.ll_solver._backtrack_solve(plan, callback=None, amax=len(plan.actions)-2)

            for action in plan.actions[:-1]:
                self.execute_plan(plan, action.active_timesteps)

            final_offset = np.array([0.0,0.0])
            for offset in utils.wrist_im_offsets:
                cloth_present = self.cloth_predictor.predict_wrist_center(offset[0])
                if cloth_present:
                    plan.params['baxter'].openrave_body.set_pose([0,0,utils.regions[0][0]])
                    accept = len(plan.params['baxter'].openrave_body.get_ik_from_pose(np.r_[cloth_pos[:2], 0.725]+np.r_[offset[1], 0]+np.r_[final_offset, 0], [0, np.pi/2, 0], "left_arm")) > 0
                    if accept:
                        final_offset += np.array(offset[1])

            cloth_pos[:2] += final_offset

            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 ROBOT_INIT_POSE CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 LOAD_WASHER_INTERMEDIATE_POSE CLOTH0 \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': cloth_pos})
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            dropped_cloth = False
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)
            else:
                continue
            
            if self.state.left_hand_range > 1:
                self.left_grip.open()
                dropped_cloth = True

            self.left_grip.close()
            self.right_grip.close()
            while not dropped_cloth:
                act_num = 0
                ll_plan_str = []

                ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_INIT_POSE PUT_INTO_WASHER_BEGIN CLOTH0 \n'.format(act_num))
                act_num += 1
                ll_plan_str.append('{0}: PUT_INTO_WASHER BAXTER WASHER WASHER_OPEN_POSE_0 CLOTH0 CLOTH_TARGET_END_0 PUT_INTO_WASHER_BEGIN CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
                act_num += 1
                ll_plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_0 INTERMEDIATE_UNLOAD \n'.format(act_num))
                act_num += 1
                ll_plan_str.append('{0}: MOVETO BAXTER INTERMEDIATE_UNLOAD IN_WASHER_ADJUST \n'.format(act_num))
                act_num += 1
                ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER IN_WASHER_ADJUST IN_WASHER_ADJUST_2 CLOTH0 \n'.format(act_num))
                act_num += 1
                ll_plan_str.append('{0}: MOVETO BAXTER IN_WASHER_ADJUST_2 INTERMEDIATE_UNLOAD \n'.format(act_num))
                act_num += 1
                ll_plan_str.append('{0}: MOVETO BAXTER INTERMEDIATE_UNLOAD LOAD_WASHER_INTERMEDIATE_POSE \n'.format(act_num))
                act_num += 1

                plan = self.plan_from_str(ll_plan_str)
                self.update_plan(plan)
                success = self.ll_solver.backtrack_solve(plan, callback=None)

                dropped_cloth = True
                if success:
                    for action in plan.actions:
                        self.execute_plan(plan, action.active_timesteps)

                if self.state.left_hand_range < 1:
                    self.left_grip.close()
                    dropped_cloth = False

                act_num = 0
                ll_plan_str = []
            self.predict_cloth_locations()

        return []

    def clear_basket_near_loc(self):
        self.predict_basket_location()
        self.predict_cloth_locations()

        failed_constr = []
        if not len(self.state.region_poses[4]) and not len(self.state.region_poses[5]):
             failed_constr.append(("BasketNearLocClear", True))

        if self.state.washer_door > -np.pi/3:
            failed_constr.append(("WasherDoorOpen", False))

        if np.all(np.abs(self.state.basket_pose[:2] - utils.basket_near_pos[:2]) < 0.15):
            failed_constr.append(("BasketInNearLoc", True))

        if np.all(np.abs(self.state.basket_pose[:2] - utils.basket_far_pos[:2]) > 0.15):
            failed_constr.append(("BasketInFarLoc", False))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        while len(self.state.region_poses[4]):
            act_num = 0
            ll_plan_str = []

            last_pose = 'ROBOT_INIT_POSE'
            if (self.state.robot_region != 1):
                ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                last_pose = 'ROBOT_REGION_1_POSE_0'

            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER {1} LOAD_BASKET_NEAR \n'.format(act_num, last_pose))
            act_num += 1
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER LOAD_BASKET_NEAR CLOTH_GRASP_BEGIN_0 \n'.format(act_num, last_pose))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))

            self.predict_cloth_locations()
            plan = self.plan_from_str(ll_plan_str)
            cloth_pos = self.get_next_cloth_pos(5)
            self.update_plan(plan, {'cloth0': cloth_pos})
            self.ll_solver._backtrack_solve(plan, callback=None, amax=len(plan.actions)-2)

            for action in plan.actions[:-1]:
                self.execute_plan(plan, action.active_timesteps)

            final_offset = np.array([0.0,0.0])
            for offset in utils.wrist_im_offsets:
                cloth_present = self.cloth_predictor.predict_wrist_center(offset[0])
                if cloth_present:
                    plan.params['baxter'].openrave_body.set_pose([0,0,utils.regions[0][0]])
                    cur_cloth_pos = np.r_[cloth_pos[:2], 0.725]+np.r_[offset[1], 0.0]+np.r_[final_offset, 0.0]
                    accept = len(plan.params['baxter'].openrave_body.get_ik_from_pose(cur_cloth_pos, [0, np.pi/2, 0], "left_arm")) > 0
                    if accept:
                        final_offset += np.array(offset[1])

            cloth_pos[:2] += final_offset

            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 ROBOT_INIT_POSE CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 LOAD_WASHER_INTERMEDIATE_POSE CLOTH0 \n'.format(act_num))
            act_num += 1
            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': cloth_pos})
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            dropped_cloth = False
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)
            else:
                continue

            if self.state.left_hand_range > 1:
                self.left_grip.open()
                dropped_cloth = True
                continue

            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_INIT_POSE PUT_INTO_WASHER_BEGIN CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_WASHER BAXTER WASHER WASHER_OPEN_POSE_0 CLOTH0 CLOTH_TARGET_END_0 PUT_INTO_WASHER_BEGIN CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_0 INTERMEDIATE_UNLOAD \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVETO BAXTER INTERMEDIATE_UNLOAD IN_WASHER_ADJUST \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER IN_WASHER_ADJUST IN_WASHER_ADJUST_2 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVETO BAXTER IN_WASHER_ADJUST_2 INTERMEDIATE_UNLOAD \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVETO BAXTER INTERMEDIATE_UNLOAD LOAD_WASHER_INTERMEDIATE_POSE \n'.format(act_num))
            act_num += 1
            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': cloth_pos})
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)

            act_num = 0
            ll_plan_str = []
            self.predict_cloth_locations()

        while len(self.state.region_poses[5]):
            act_num = 0
            ll_plan_str = []

            last_pose = 'ROBOT_INIT_POSE'

            if (self.state.robot_region != 2):
                ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE ARMS_FORWARD_{1} \n'.format(act_num, self.state.robot_region))
                act_num += 1
                ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ARMS_FORWARD_{1} ROBOT_REGION_{2}_POSE_0 \n'.format(act_num, self.state.robot_region, self.state.robot_region))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_2_POSE_0 REGION2 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                last_pose = 'ROBOT_REGION_2_POSE_0'

            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER {1} CLOTH_GRASP_BEGIN_0 \n'.format(act_num, last_pose))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1


            plan = self.plan_from_str(ll_plan_str)
            cloth_pos = self.get_next_cloth_pos(6)
            self.update_plan(plan, {'cloth0': cloth_pos})
            self.ll_solver._backtrack_solve(plan, callback=None, amax=len(plan.actions)-2)

            for action in plan.actions[:-1]:
                self.execute_plan(plan, action.active_timesteps)

            final_offset = np.array([0.0,0.0])
            for offset in utils.wrist_im_offsets:
                cloth_present = self.cloth_predictor.predict_wrist_center(offset[0])
                if cloth_present:
                    plan.params['baxter'].openrave_body.set_pose([0,0,utils.regions[1][0]])
                    accept = len(plan.params['baxter'].openrave_body.get_ik_from_pose(np.r_[cloth_pos[:2], 0.725]+np.r_[offset[1], 0]+np.r_[final_offset, 0], [0, np.pi/2, 0], "left_arm")) > 0
                    if accept:
                        final_offset += np.array(offset[1])
                        break

            cloth_pos[:2] += final_offset

            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 ROBOT_INIT_POSE CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 LOAD_BASKET_INTER_2 CLOTH0 \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': cloth_pos})
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            dropped_cloth = False
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)
            else:
                continue

            if self.state.left_hand_range > 1:
                self.left_grip.open()
                dropped_cloth = True
                continue

            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: ROTATE_HOLDING_CLOTH BAXTER CLOTH0 ROBOT_INIT_POSE ROBOT_REGION_3_POSE_0 REGION3 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_REGION_3_POSE_0 LOAD_BASKET_FAR CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER LOAD_BASKET_FAR CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_FAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1
            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan)
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            dropped_cloth = False
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)
                    if action in plan.actions[:1] and self.state.left_hand_range > 1:
                        self.left_grip.open()
                        dropped_cloth = True
                        break

            act_num = 0
            ll_plan_str = []
            self.predict_basket_location()
            self.predict_cloth_locations()

        return []

    def clear_basket_far_loc(self):
        self.predict_basket_location()
        self.predict_cloth_locations()

        failed_constr = []
        if not len(self.state.region_poses[5]):
             failed_constr.append(("BasketFarLocClear", True))

        if np.all(np.abs(self.state.basket_pose - utils.basket_far_pos)[:2] < 0.15):
            failed_constr.append(("BasketInFarLoc", True))

        if np.any(np.abs(self.state.basket_pose - utils.basket_near_pos)[:2] > 0.15):
            failed_constr.append(("BasketInNearLoc", False))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        while len(self.state.region_poses[6]):
            start_pose = 'ROBOT_INIT_POSE'
            if (self.state.robot_region != 3):
                ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_3_POSE_0 REGION3 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                start_pose = 'ROBOT_REGION_3_POSE_0'
            ll_plan_str.append('{0}: MOVETO BAXTER {1} CLOTH_GRASP_BEGIN_0 \n'.format(act_num, start_pose))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE_HOLDING_CLOTH BAXTER CLOTH0 CLOTH_GRASP_END_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_REGION_1_POSE_0 LOAD_BASKET_NEAR CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER LOAD_BASKET_NEAR CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            cloth_pos = self.get_next_cloth_pos(7)
            self.update_plan(plan, {'cloth0': cloth_pos})
            self.ll_solver.backtrack_solve(plan, callback=None)

            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

            act_num = 0
            ll_plan_str = []
            self.predict_cloth_locations()

        return []

    def unload_washer_into_basket(self):
        self.predict_basket_location()
        # self.predict_cloth_locations()

        failed_constr = []
        if self.state.washer_door > -np.pi/3:
            failed_constr.append(("WasherDoorOpen", False))

        if np.any(np.abs(self.state.basket_pose - utils.basket_near_pos)[:2] > 0.15):
            failed_constr.append(("BasketInNearLoc", False))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        last_pose = 'ROBOT_INIT_POSE'
        if (self.state.robot_region != 1):
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            last_pose = 'ROBOT_REGION_1_POSE_0'

        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER {1} LOAD_BASKET_NEAR \n'.format(act_num, last_pose))
        act_num += 1
        ll_plan_str.append('{0}: MOVETO BAXTER LOAD_BASKET_NEAR LOAD_WASHER_INTERMEDIATE_POSE \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVETO BAXTER LOAD_WASHER_INTERMEDIATE_POSE WASHER_SCAN_POSE \n'.format(act_num, last_pose))
        act_num += 1
        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan)
        self.ll_solver.backtrack_solve(plan, callback=None)
        for action in plan.actions:
            self.execute_plan(plan, action.active_timesteps)

        while self.cloth_predictor.predict_wrist_center(offset=[75, -30]):
            ll_plan_str = []
            act_num = 0
            ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE INTERMEDIATE_UNLOAD \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVETO BAXTER INTERMEDIATE_UNLOAD UNLOAD_WASHER_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER UNLOAD_WASHER_0 INTERMEDIATE_UNLOAD CLOTH0'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER INTERMEDIATE_UNLOAD LOAD_WASHER_INTERMEDIATE_POSE CLOTH0'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, in_washer=True)
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            dropped_cloth = False
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)

                if self.state.left_hand_range > 1:
                    self.left_grip.open()
                    dropped_cloth = True

                    act_num = 0
                    ll_plan_str = []
                    ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE WASHER_SCAN_POSE \n'.format(act_num))
                    plan = self.plan_from_str(ll_plan_str)
                    self.update_plan(plan)
                    self.ll_solver.backtrack_solve(plan, callback=None)
                    for action in plan.actions:
                        self.execute_plan(plan, action.active_timesteps)

                    continue

            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_INIT_POSE LOAD_BASKET_NEAR CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER LOAD_BASKET_NEAR CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_1 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER CLOTH_PUTDOWN_END_0 LOAD_WASHER_INTERMEDIATE_POSE \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVETO BAXTER LOAD_WASHER_INTERMEDIATE_POSE WASHER_SCAN_POSE \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, in_washer=True)
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            dropped_cloth = False
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)

            act_num = 0
            ll_plan_str = []

        while self.cloth_predictor.predict_wrist_center(offset=[-30, 0]):
            ll_plan_str = []
            act_num = 0
            ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE UNLOAD_WASHER_3 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER UNLOAD_WASHER_3 INTERMEDIATE_UNLOAD CLOTH0'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER INTERMEDIATE_UNLOAD LOAD_WASHER_INTERMEDIATE_POSE CLOTH0'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, in_washer=True)
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            dropped_cloth = False
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)

                if self.state.left_hand_range > 1:
                    self.left_grip.open()
                    dropped_cloth = True

                    act_num = 0
                    ll_plan_str = []
                    ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE WASHER_SCAN_POSE \n'.format(act_num))
                    plan = self.plan_from_str(ll_plan_str)
                    self.update_plan(plan)
                    self.ll_solver.backtrack_solve(plan, callback=None)
                    for action in plan.actions:
                        self.execute_plan(plan, action.active_timesteps)

                    continue

            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_INIT_POSE LOAD_BASKET_NEAR CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER LOAD_BASKET_NEAR CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_1 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_0 LOAD_WASHER_INTERMEDIATE_POSE \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVETO BAXTER LOAD_WASHER_INTERMEDIATE_POSE WASHER_SCAN_POSE \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, in_washer=True)
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            dropped_cloth = False
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)

            act_num = 0
            ll_plan_str = []
            self.predict_cloth_washer_locations()

        act_num = 0
        ll_plan_str = []
        ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE INTERMEDIATE_UNLOAD \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVETO BAXTER INTERMEDIATE_UNLOAD LOAD_WASHER_INTERMEDIATE_POSE \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER LOAD_WASHER_INTERMEDIATE_POSE LOAD_BASKET_NEAR \n'.format(act_num))
        act_num += 1

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, in_washer=True)
        self.ll_solver.backtrack_solve(plan, callback=None)

        for action in plan.actions:
            self.execute_plan(plan, action.active_timesteps)

        return []

    def reset_laundry(self):
        '''
        Used to have Baxter reset its environment to a good initial configuration.
        '''
        valid_regions = [1, 2, 3, 5]
        self.predict_basket_location()

        if self.state.washer_door > -np.pi/6:
            self.open_washer()

        if np.any(np.abs(self.state.basket_pose[:2] - utils.basket_near_pos[:2]) > 0.2):
            self.move_basket_to_washer()

        self.unload_washer_into_basket()
        self.load_basket_in_region_1()
        self.move_basket_from_washer()

        self.predict_basket_location()
        self.predict_cloth_locations()
        while len(self.state.region_poses[6]):
            next_region_ind = np.random.choice(range(len(utils.cloth_grid_coordinates)))
            while utils.cloth_grid_coordinates[next_region_ind][1] not in valid_regions:
                next_region_ind = np.random.choice(range(len(utils.cloth_grid_coordinates)))

            end_pose = utils.cloth_grid_coordinates[next_region_ind][0]
            print end_pose
            next_region = utils.cloth_grid_coordinates[next_region_ind][1] % 4

            act_num = 0
            ll_plan_str = []

            last_pose = 'ROBOT_INIT_POSE'
            if (self.state.robot_region != 3):
                ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_3_POSE_0 REGION3 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                last_pose = 'ROBOT_REGION_3_POSE_0'

            ll_plan_str.append('{0}: MOVETO BAXTER {1} CLOTH_GRASP_BEGIN_0 \n'.format(act_num, last_pose))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 LOAD_BASKET_FAR CLOTH0'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            next_pose = self.get_next_cloth_pos(7)
            self.update_plan(plan, {'cloth0':next_pose}, cloth_z=0.635)
            success = self.ll_solver.backtrack_solve(plan, callback=None, n_resamples=3)
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)

            if self.state.left_hand_range > 1:
                self.left_grip.open()
                continue
            
            act_num = 0
            ll_plan_str = []

            if next_region != 3:
                ll_plan_str.append('{0}: ROTATE_HOLDING_CLOTH BAXTER CLOTH0 ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_1 REGION{2}'.format(act_num, next_region, next_region))
                act_num += 1
                ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_REGION_{1}_POSE_1 CLOTH_PUTDOWN_BEGIN_0 CLOTH0'.format(act_num, next_region))
                act_num += 1
            else:
                ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER LOAD_BASKET_FAR CLOTH_PUTDOWN_BEGIN_0 CLOTH0'.format(act_num))
                act_num += 1
            ll_plan_str.append('{0}: CLOTH_PUTDOWN BAXTER CLOTH0 CLOTH_TARGET_END_0 CLOTH_PUTDOWN_BEGIN_0 CP_EE_1 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_0 ARMS_FORWARD_{1} \n'.format(act_num, next_region))
            act_num += 1
            ll_plan_str.append('{0}: MOVE_AROUND_WASHER BAXTER ARMS_FORWARD_{1} BASKET_SCAN_POSE_{2} \n'.format(act_num, next_region, next_region))
            act_num += 1


            plan = self.plan_from_str(ll_plan_str)
            next_pose = self.get_next_cloth_pos(7)
            self.update_plan(plan)
            plan.params['cloth_target_end_0'].value[:2,0] = end_pose + np.array([0.03, 0])
            plan.params['cloth_target_end_0'].value[2,0] = 0.645
            plan.params['cloth_target_end_0']._free_attrs['value'][:] = 0
            success = self.ll_solver.backtrack_solve(plan, callback=None, n_resamples=3)
            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)
            plan.params['cloth_target_end_0']._free_attrs['value'][:] = 1
            self.predict_cloth_locations()

        self.close_washer()

        return []

    def fold_cloth(self):
        self.predict_cloth_locations()

        failed_constr = []
        if not len(self.state.region_poses[1]):
             failed_constr.append(("ClothInRegion2", False))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        start_pose = 'ROBOT_INIT_POSE'
        if (self.state.robot_region != 2):
            ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_2_POSE_0 REGION2 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            start_pose = 'ROBOT_REGION_2_POSE_0'
        ll_plan_str.append('{0}: MOVETO BAXTER {1} START_GRASP_2 \n'.format(act_num, start_pose))
        act_num += 1
        ll_plan_str.append('{0}: MOVETO BAXTER START_GRASP_2 CLOTH_GRASP_BEGIN_0 \n'.format(act_num, start_pose))
        act_num += 1
        ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
        act_num += 1

        cloth_pos = self.get_next_cloth_pos(2)
        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {'cloth0': cloth_pos})
        success = self.ll_solver._backtrack_solve(plan, callback=None, amax=len(plan.actions)-2)

        if success:
            for action in plan.actions[:-1]:
                self.execute_plan(plan, action.active_timesteps)

        final_offset = np.array([0.0,0.0])
        for offset in utils.wrist_im_offsets:
            cloth_present = self.cloth_predictor.predict_wrist_center(offset[0])
            if cloth_present:
                plan.params['baxter'].openrave_body.set_pose([0,0,utils.regions[1][0]])
                accept = len(plan.params['baxter'].openrave_body.get_ik_from_pose(np.r_[cloth_pos[:2], 0.725]+np.r_[offset[1], 0]+np.r_[final_offset, 0], [0, np.pi/2, 0], "left_arm")) > 0
                if accept:
                    final_offset += np.array(offset[1])

        cloth_pos[:2] += final_offset

        self.state.robot_region = 2

        for i in range(1):
            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 LOAD_BASKET_INTER_2 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER LOAD_BASKET_INTER_2 FOLD_AFTER_GRASP CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER FOLD_AFTER_GRASP FOLD_AFTER_DRAG CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVETO BAXTER FOLD_AFTER_DRAG FOLD_SCAN_CORNER_1 \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)

            self.update_plan(plan, {'cloth0': cloth_pos})
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)
            cloth_pos[:2] = plan.params['cloth0'].pose[:2,-1] + np.array([0.05, -0.2])


        for i in range(1):
            act_num = 0
            ll_plan_str = []

            ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 FOLD_AFTER_GRASP_2 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER FOLD_AFTER_GRASP_2 FOLD_AFTER_DRAG CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVETO BAXTER FOLD_AFTER_DRAG FOLD_SCAN_CORNER_1 \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)

            self.update_plan(plan, {'cloth0': cloth_pos})
            success = self.ll_solver.backtrack_solve(plan, callback=None)

            if success:
                for action in plan.actions:
                    self.execute_plan(plan, action.active_timesteps)

            cloth_offset = self.corner_predictor.predict("left")
            left_joint_dict = self.left_arm.joint_angles()
            left_joints = map(lambda j: left_joint_dict[j], const.left_joints)
            plan.params['baxter'].openrave_body.set_pose([0,0,utils.regions[1][0]])
            plan.params['baxter'].openrave_body.set_dof({'lArmPose': left_joints})
            ee_pos = plan.params['baxter'].openrave_body.env_body.GetLink("left_gripper").GetTransform()[:3,3]

            cloth_pos[:2] = ee_pos[:2] + cloth_offset

        for i in range(5):
            cloth_offset = self.corner_predictor.predict("left")
            if np.any(np.abs(cloth_offset) > 0.02):
                left_joint_dict = self.left_arm.joint_angles()
                left_joints = map(lambda j: left_joint_dict[j], const.left_joints)
                plan.params['baxter'].openrave_body.set_pose([0,0,utils.regions[1][0]])
                plan.params['baxter'].openrave_body.set_dof({'lArmPose': left_joints})
                ee_pos = plan.params['baxter'].openrave_body.env_body.GetLink("left_gripper").GetTransform()[:3,3]

                cloth_pos[:2] = ee_pos[:2] + cloth_offset + [0.04, 0.04]

                lposes = plan.params['baxter'].openrave_body.get_ik_from_pose(np.r_[cloth_pos, 1.05], [0, np.pi/2, 0], "left_arm")
                if not len(lposes):
                    cloth_pos[:2] = ee_pos[:2] + cloth_offset / 2.0
                    lposes = plan.params['baxter'].openrave_body.get_ik_from_pose(np.r_[cloth_pos, 1.05], [0, np.pi/2, 0], "left_arm")
                    if not len(lposes):
                        break

                pose_ind = np.argmin(np.sum(np.abs(lposes - left_joints), axis=1))
                pose = lposes[pose_ind]

                act_num = 0
                ll_plan_str = []

                ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE FOLD_SCAN_CORNER_1 \n'.format(act_num))
                act_num += 1

                plan = self.plan_from_str(ll_plan_str)
                plan.params['fold_scan_corner_1'].lArmPose[:,0] = pose

                self.update_plan(plan)
                success = self.ll_solver.backtrack_solve(plan, callback=None)

                if success:
                    for action in plan.actions:
                        self.execute_plan(plan, action.active_timesteps)
            else:
                break

        act_num = 0
        ll_plan_str = []

        ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 FOLD_AFTER_GRASP_2 CLOTH0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER FOLD_AFTER_GRASP_2 FOLD_AFTER_DRAG_2 CLOTH0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVETO BAXTER FOLD_AFTER_DRAG_2 FOLD_SCAN_CORNER_2 \n'.format(act_num))
        act_num += 1

        plan = self.plan_from_str(ll_plan_str)

        self.update_plan(plan, {'cloth0': cloth_pos})
        success = self.ll_solver.backtrack_solve(plan, callback=None)

        if success:
            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

        for i in range(5):
            cloth_offset = self.corner_predictor.predict("right")
            if np.any(np.abs(cloth_offset) > 0.02):
                right_joint_dict = self.right_arm.joint_angles()
                right_joints = map(lambda j: right_joint_dict[j], const.right_joints)
                left_joint_dict = self.left_arm.joint_angles()
                left_joints = map(lambda j: left_joint_dict[j], const.left_joints)
                plan.params['baxter'].openrave_body.set_pose([0,0,utils.regions[1][0]])
                plan.params['baxter'].openrave_body.set_dof({'rArmPose': right_joints})
                ee_pos = plan.params['baxter'].openrave_body.env_body.GetLink("right_gripper").GetTransform()[:3,3]

                cloth_pos[:2] = ee_pos[:2] + cloth_offset + [0.04, 0.02]

                rposes = plan.params['baxter'].openrave_body.get_ik_from_pose(np.r_[cloth_pos+[0, 0.03], 1.05], [0, 31*np.pi/64, 0], "right_arm")
                if not len(rposes):
                    cloth_pos[:2] = ee_pos[:2] + cloth_offset / 2.0 + [0.04, 0.02]
                    rposes = plan.params['baxter'].openrave_body.get_ik_from_pose(np.r_[cloth_pos, 1.05], [0, 31*np.pi/64, 0], "right_arm")
                    if not len(rposes):
                        break

                pose_ind = np.argmin(np.sum(np.abs(rposes - right_joints), axis=1))
                pose = rposes[pose_ind]

                act_num = 0
                ll_plan_str = []

                ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE FOLD_SCAN_CORNER_2 \n'.format(act_num))
                act_num += 1

                plan = self.plan_from_str(ll_plan_str)
                plan.params['fold_scan_corner_2'].rArmPose[:,0] = pose
                plan.params['fold_scan_corner_2'].lArmPose[:,0] = left_joints

                self.update_plan(plan)
                success = self.ll_solver.backtrack_solve(plan, callback=None)

                if success:
                    for action in plan.actions:
                        self.execute_plan(plan, action.active_timesteps)
            else:
                break

        right_joint_dict = self.right_arm.joint_angles()
        right_joints = map(lambda j: right_joint_dict[j], const.right_joints)
        left_joint_dict = self.left_arm.joint_angles()
        left_joints = map(lambda j: left_joint_dict[j], const.left_joints)
        plan.params['baxter'].openrave_body.set_pose([0,0,utils.regions[1][0]])
        plan.params['baxter'].openrave_body.set_dof({'rArmPose': right_joints, 'lArmPose': left_joints})
        left_corner = plan.params['baxter'].openrave_body.env_body.GetLink("left_gripper").GetTransform()[:2,3]
        right_corner = (ee_pos[:2] + cloth_offset + [0, 0.02]).flatten()

        center = (left_corner + right_corner) / 2
        rotation = (right_corner[0] - left_corner[0]) / np.arcsin(np.sqrt(np.sum((left_corner - right_corner)**2)))

        act_num = 0
        ll_plan_str = []

        ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BOTH_END_CLOTH_GRASP BAXTER CLOTH_LONG_EDGE CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CG_EE_1 CLOTH_GRASP_END_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BOTH_MOVE_CLOTH_TO BAXTER CLOTH_LONG_EDGE CLOTH_FOLD_AIR_TARGET_1 CLOTH_GRASP_END_0 CLOTH_GRASP_END_1 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BOTH_MOVE_CLOTH_TO BAXTER CLOTH_LONG_EDGE CLOTH_FOLD_TABLE_TARGET_1 CLOTH_GRASP_END_1 CLOTH_GRASP_END_2 \n'.format(act_num))
        act_num += 1

        plan = self.plan_from_str(ll_plan_str)
        plan.params['cloth_long_edge'].pose[:,0] = np.r_[center, 0.64]
        plan.params['cloth_long_edge'].rotation[:,0] = [rotation, 0, -np.pi/2]
        plan.params['cloth_target_begin_0'].value[:,0] = np.r_[center, 0.64]
        plan.params['cloth_target_begin_0'].rotation = np.array([[rotation], [0], [-np.pi/2]])
        plan.params['cloth_target_begin_0']._free_attrs['value'][:] = 0
        plan.params['cloth_target_begin_0']._free_attrs['rotation'][:] = 0

        self.update_plan(plan)
        success = self.ll_solver.backtrack_solve(plan, callback=None)

        if success:
            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

        right_joint_dict = self.right_arm.joint_angles()
        right_joints = map(lambda j: right_joint_dict[j], const.right_joints)
        left_joint_dict = self.left_arm.joint_angles()
        left_joints = map(lambda j: left_joint_dict[j], const.left_joints)
        plan.params['baxter'].openrave_body.set_pose([0,0,utils.regions[1][0]])
        plan.params['baxter'].openrave_body.set_dof({'rArmPose': right_joints, 'lArmPose': left_joints})
        left_corner = [0.5, 0.35, 0.625]
        right_corner = [0.5, -0.35, 0.625]

        act_num = 0
        ll_plan_str = []

        ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE FOLD_SCAN_TWO_CORNER \n'.format(act_num))
        act_num += 1

        plan = self.plan_from_str(ll_plan_str)
        plan.params['fold_scan_two_corner'].rArmPose[:,0] = right_joints

        self.update_plan(plan)
        plan.params['cloth0'].openrave_body.set_pose([1,1,1])
        success = self.ll_solver.backtrack_solve(plan, callback=None)

        if success:
            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

        act_num = 0
        ll_plan_str = []

        ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE CLOTH_GRASP_BEGIN_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BOTH_END_CLOTH_GRASP BAXTER CLOTH_SHORT_EDGE CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CG_EE_1 CLOTH_GRASP_END_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BOTH_MOVE_CLOTH_TO BAXTER CLOTH_SHORT_EDGE CLOTH_FOLD_AIR_TARGET_2 CLOTH_GRASP_END_0 CLOTH_GRASP_END_1 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BOTH_MOVE_CLOTH_TO BAXTER CLOTH_SHORT_EDGE CLOTH_FOLD_TABLE_TARGET_2 CLOTH_GRASP_END_1 CLOTH_GRASP_END_2 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BOTH_MOVE_CLOTH_TO BAXTER CLOTH_SHORT_EDGE CLOTH_FOLD_TABLE_TARGET_3 CLOTH_GRASP_END_2 CLOTH_GRASP_END_3 \n'.format(act_num))
        act_num += 1

        plan = self.plan_from_str(ll_plan_str)

        cloth_offset = self.corner_predictor.predict("left")
        self.update_plan(plan)
        plan.params['cloth0'].openrave_body.set_pose([1,1,1])
        if cloth_offset[1] > 0:
            plan.params['cloth_long_edge'].pose[:,0] = [0.6, 0.25, 0.64]
            plan.params['cloth_long_edge'].rotation[:,0] = [np.pi/4, 0, -np.pi/2]
            plan.params['cloth_target_begin_0'].value[:,0] = [0.65, 0.25, 0.64]
            plan.params['cloth_target_begin_0'].rotation = np.array([[np.pi/4], [0], [-np.pi/2]])
            plan.params['cloth_target_begin_0']._free_attrs['value'][:] = 0
            plan.params['cloth_target_begin_0']._free_attrs['rotation'][:] = 0
        else:
            plan.params['cloth_long_edge'].pose[:,0] = [0.6, -0.25, 0.64]
            plan.params['cloth_long_edge'].rotation[:,0] = [-np.pi/4, 0, -np.pi/2]
            plan.params['cloth_target_begin_0'].value[:,0] = [0.65, -0.25, 0.64]
            plan.params['cloth_target_begin_0'].rotation = np.array([[-np.pi/4], [0], [-np.pi/2]])
            plan.params['cloth_target_begin_0']._free_attrs['value'][:] = 0
            plan.params['cloth_target_begin_0']._free_attrs['rotation'][:] = 0

        success = self.ll_solver.backtrack_solve(plan, callback=None)

        if success:
            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

        return []

