import copy
import main

import numpy as np

import rospy

import baxter_interface

from core.parsing import parse_domain_config, parse_problem_config
import core.util_classes.baxter_constants as const
from core.util_classes.viewer import OpenRAVEViewer
from pma.hl_solver import FFSolver
from pma.robot_ll_solver import RobotLLSolver
from ros_interface.basket.basket_predict import BasketPredict
from ros_interface.basket_wrist.basket_wrist_predict import BasketWristPredict
from ros_interface.cloth.cloth_grid_predict import ClothGridPredict
from ros_interface.controllers import EEController, TrajectoryController
from ros_interface.rotate_control import RotateControl
import ros_interface.utils as utils


DOMAIN_FILE = "../domains/laundry_domain/domain.pddl"

class HLLaundryState(object):
    # TODO: This class is here for temporary usage but needs generalization
    #       Ideally integreate with existing HLState class and Env Monitor
    def __init__(self):
        self.region_poses = [[], [], [], [], [], []]
        self.basket_pose = [utils.basket_far_pos[0], utils.basket_far_pos[1], utils.basket_far_rot[0]]
        self.washer_cloth_poses = []

        self.robot_region = 1
        self.washer_door = 0

        # For constructing high level plans
        self.prob_domain = "(:domain laundry_domain)\n"
        self.objects = "(:objects cloth washer basket)\n"
        self.goal = "(:goal (and (ClothInWasher cloth washer) (not (WasherDoorOpen washer)))\n"

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
        self.region_poses = [[], [], [], [], [], []]
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
        else:
            self.hl_preds["BasketNearLocClear"] = True

        if len(self.region_poses[5]):
            cloth_in_washer = False
            if self.hl_preds["BasketInFarLoc"]:
                self.hl_preds["ClothInBasket"] = True
            else:
                self.hl_preds["BasketFarLocClear"] = False
        else:
            self.hl_preds["BasketFarLocClear"] = True

        if (not (self.hl_preds["BasketInNearLoc"] and len(self.region_poses[4]))) and (not (self.hl_preds["BasketInFarLoc"] and len(self.region_poses[5]))):
            self.hl_preds["ClothInBasket"] = False

        self.hl_preds["ClothInWasher"] = cloth_in_washer


    def store_washer_poses(self, poses):
        self.washer_cloth_poses = poses.deepcopy()
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
        elif np.all(np.abs(self.basket_pose[:2] - utils.basket_far_pos[:2]) < 0.15) and np.abs(self.basket_pose[2] - utils.basket_far_rot[0]) < 0.5:
            self.hl_preds["BasketInFarLoc"] = True

    def update_door(self, open):
        self.hl_preds["WasherDoorOpen"] = open


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
        self.cloth_predictor = ClothGridPredict()
        self.basket_predictor = BasketPredict()
        # self.basket_wrist_predictor = BasketWristPredict()
        # self.ee_control = EEController()
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

    def run_baxter(self):
        success = False
        while not success:
            success = True
            self.predict_cloth_locations()
            self.predict_basket_location()
            hl_plan = self.solve_hl_prob()
            if hl_plan == "Impossible":
                print "Impossible Plan"
            import ipdb; ipdb.set_trace()
            for action in hl_plan:
                failed = []
                act_type = action.split()[1].lower()
                if act_type == "load_basket_from_region_1":
                    failed = self.load_basket_from_region_1()
                elif act_type == "load_basket_from_region_2":
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
                else:
                    failed = []

                import ipdb; ipdb.set_trace()

                if len(failed):
                    self.state.update(failed)
                    success = False
                    break

        import ipdb; ipdb.set_trace()

    def solve_hl_prob(self):
        abs_prob = self.state.get_abs_prob()
        return self.hl_solver._run_planner(self.abs_domain, abs_prob)

    def execute_plan(self, plan, active_ts):
        current_ts = active_ts[0]
        while (current_ts < active_ts[1] and current_ts < plan.horizon):
            cur_action = filter(lambda a: a.active_timesteps[0] == current_ts, plan.actions)[0]
            if cur_action.name == "open_door":
                self.state.washer_door = -np.pi/2
            elif cur_action.name == "close_door":
                self.state.washer_door = 0

            if cur_action.name.startswith("rotate"):
                old_region = self.state.robot_region
                self.state.robot_region = int(cur_action.params[-1].name[-1]) # TODO: Make this more generalized
                if self.state.robot_region != old_region:
                    self.rotate_control.rotate_to_region(self.state.robot_region)
            else:
                self.traj_control.execute_plan(plan, active_ts=cur_action.active_timesteps)
            current_ts = cur_action.active_timesteps[1]

    def predict_cloth_locations(self):
        locs = self.cloth_predictor.predict()
        self.state.store_cloth_regions(locs)

    def predict_basket_location(self):
        act_num = 0
        ll_plan_str = []
        if (self.state.robot_region != 3):
            ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE BASKET_SCAN_POSE_{1} \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER BASKET_SCAN_POSE_{1} BASKET_SCAN_POSE_2 REGION2 \n'.format(act_num, self.state.robot_region))
            act_num += 1

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        self.ll_solver._backtrack_solve(plan, callback=None)

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

    def update_plan(self, plan, cloth_to_region={}, in_washer=False, reset_to_region=-1):
        plan.params['basket'].pose[:2, 0] = self.state.basket_pose[:2]
        plan.params['basket'].rotation[0, 0] = self.state.basket_pose[2]

        left_joint_dict = self.left_arm.joint_angles()
        left_joints = map(lambda j: left_joint_dict[j], const.left_joints) # [left_joint_dict[joint] for joint in const.left_joints]
        left_grip = const.GRIPPER_OPEN_VALUE if self.left_grip.position() > 50 else const.GRIPPER_CLOSE_VALUE
        right_joint_dict = self.right_arm.joint_angles()
        right_joints = map(lambda j: right_joint_dict[j], const.right_joints) # [right_joint_dict[joint] for joint in const.right_joints]
        right_grip = const.GRIPPER_OPEN_VALUE if self.right_grip.position() > 50 else const.GRIPPER_CLOSE_VALUE
        plan.params['baxter'].lArmPose[:, 0] = left_joints
        plan.params['baxter'].lGripper[:, 0] = left_grip
        plan.params['baxter'].rArmPose[:, 0] = right_joints
        plan.params['baxter'].rGripper[:, 0] = right_grip
        plan.params['baxter'].pose[:,0] = (self.state.robot_region - 2) * -np.pi/4

        plan.params['robot_init_pose'].lArmPose[:, 0] = left_joints
        plan.params['robot_init_pose'].lGripper[:, 0] = left_grip
        plan.params['robot_init_pose'].rArmPose[:, 0] = right_joints
        plan.params['robot_init_pose'].rGripper[:, 0] = right_grip
        plan.params['robot_init_pose'].value[:,0] = (self.state.robot_region - 2) * -np.pi/4

        plan.params['washer'].door[:,0] = self.state.washer_door

        poses = copy.deepcopy(self.state.region_poses)
        if not self.state['BasketInNearLoc']:
            poses[0].extend(poses[4])
        if not self.state['BasketInFarLoc']:
            poses[2].extend(poses[5])
        for cloth in cloth_to_region:
            next_ind = np.random.choice(range(len(poses[cloth_to_region[cloth]-1])))
            next_pose = poses[cloth_to_region[cloth]-1].pop(next_ind)
            plan.params['cloth{0}'.format(cloth)].pose[:2, 0] = next_pose
            plan.params['cloth_target_begin_{0}'.format(cloth)].value[:, 0] = plan.params['cloth{0}'.format(cloth)].pose[:, 0]

        if len(self.state.washer_cloth_poses) and in_washer:
            plan.params['cloth0'].pose[:, 0] = self.state.washer_cloth_poses[0]
            plan.params['cloth_target_begin_0'].value[:, 0] = plan.params['cloth0'].pose[:, 0]

        if reset_to_region > 0:
            filter_poses = lambda p: p[1] == reset_to_region or (reset_to_region == 1 && p[1] == 5)
            possible_poses = filter(filter_poses, utils.cloth_grid_coordinates)
            next_pose = random.choice(possible_poses)
            ref = utils.cloth_grid_ref
            disp = np.array(ref[0] - next_pose[0]) / utils.pixels_per_cm
            true_pose = (ref[1] + disp) / 100.0
            plan.params['cloth_target_end_0'].value[:2,0] = true_pose

    def plan_from_str(self, ll_plan_str):
        '''Convert a plan string (low level) into a plan object.'''
        num_cloth = 1 # sum([len(poses) for poses in self.state.region_poses])
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = FFSolver(d_c)
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_{0}.prob'.format(num_cloth))
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        return hls.get_plan(ll_plan_str, domain, problem)

    def load_basket_from_region_1(self):
        self.predict_cloth_locations()
        self.predict_basket_location()

        failed_constr = []
        if not len(self.state.region_poses[0]):
             failed_constr.append(("ClothInRegion1", False))

        if np.any(np.abs(self.state.basket_pose - utils.basket_far_pos)[:2] > 0.1):
            failed_constr.append(("BasketNearWasher", True))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        if (self.state.robot_region != 1):
            ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
            act_num += 1

        while len(self.state.region_poses[0]):
            ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_1_POSE_0 CLOTH_GRASP_BEGIN_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5} \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 CLOTH_PUTDOWN_BEGIN_{2} CLOTH{3} \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH{1} BASKET CLOTH_TARGET_END_{2} BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5} \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': 1})
            self.ll_solver.backtrack_solve(plan, callback=None)

            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

            act_num = 0
            ll_plan_str = []
            self.predict_cloth_locations()
            # self.predict_basket_location()

            # if np.any(np.abs(self.state.basket_pose - utils.basket_far_pos)[:2] > 0.1):
            #     return [("BasketNearWasher", True)]

        return []


    def load_basket_from_region_2(self):
        self.predict_cloth_locations()
        self.predict_basket_location()

        failed_constr = []
        if not len(self.state.region_poses[1]):
             failed_constr.append(("ClothInRegion2", False))

        if np.any(np.abs(self.state.basket_pose - utils.basket_far_pos)[:2] > 0.1):
            failed_constr.append(("BasketNearWasher", True))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        while len(self.state.region_poses[1]):
            start_pose = 'ROBOT_INIT_POSE'
            if (self.state.robot_region != 2):
                ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_2_POSE_0 REGION2 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                start_pose = 'ROBOT_REGION_2_POSE_0'
            ll_plan_str.append('{0}: MOVETO BAXTER {1} CLOTH_GRASP_BEGIN_0 \n'.format(act_num, start_pose))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE_HOLDING_CLOTH BAXTER CLOTH0 CLOTH_GRASP_END_0 ROBOT_REGION_3_POSE_0 REGION3 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_REGION_3_POSE_0 CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_FAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': 2})
            self.ll_solver.backtrack_solve(plan, callback=None)

            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

            act_num = 0
            ll_plan_str = []
            self.predict_cloth_locations()
            # self.predict_basket_location()

            # if np.any(np.abs(self.state.basket_pose - utils.basket_far_pos) > 0.05):
            #     failed_constr.append(("BasketNearWasher", True))

        return []

    def load_basket_from_region_3(self):
        self.predict_cloth_locations()
        # self.predict_basket_location()

        failed_constr = []
        if not len(self.state.region_poses[2]):
             failed_constr.append(("ClothInRegion3", False))

        if np.any(np.abs(self.state.basket_pose - utils.basket_far_pos)[:2] > 0.1):
            failed_constr.append(("BasketNearWasher", True))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        if (self.state.robot_region != 3):
            ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_3_POSE_0 REGION3 \n'.format(act_num, self.state.robot_region))
            act_num += 1

        while len(self.state.region_poses[1]):
            ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_3_POSE_0 CLOTH_GRASP_BEGIN_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0} CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_FAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0} CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': 3})
            self.ll_solver.backtrack_solve(plan, callback=None)

            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

            act_num = 0
            ll_plan_str = []
            self.predict_cloth_locations()
            self.predict_basket_location()

            # if np.any(np.abs(self.state.basket_pose - utils.basket_far_pos) > 0.05):
            #     failed_constr.append(("BasketNearWasher", True))

        return []

    def load_basket_from_region_4(self):
        self.predict_cloth_locations()
        self.predict_basket_location()

        failed_constr = []
        if not len(self.state.region_poses[3]):
             failed_constr.append(("ClothInRegion4", False))

        if np.any(np.abs(self.state.basket_pose - utils.basket_far_pos)[:2] > 0.1):
            failed_constr.append(("BasketNearWasher", True))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        while len(self.state.region_poses[1]):
            start_pose = 'ROBOT_INIT_POSE'
            if (self.state.robot_region != 4):
                ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_4_POSE_0 REGION4 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                start_pose = 'ROBOT_REGION_4_POSE_0'
            ll_plan_str.append('{0}: MOVETO BAXTER {1} CLOTH_GRASP_BEGIN_0 \n'.format(act_num, start_pose))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE_HOLDING_CLOTH BAXTER CLOTH0 CLOTH_GRASP_END_0 ROBOT_REGION_3_POSE_0 REGION3 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_REGION_3_POSE_0 CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_FAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': 4})
            self.ll_solver.backtrack_solve(plan, callback=None)

            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

            act_num = 0
            ll_plan_str = []
            self.predict_cloth_locations()
            # self.predict_basket_location()

            # if np.any(np.abs(self.state.basket_pose - utils.basket_far_pos) > 0.05):
            #     failed_constr.append(("BasketNearWasher", True))

        return []

    def move_basket_to_washer(self):
        self.predict_cloth_locations()
        self.predict_basket_location()

        failed_constr = []
        if len(self.state.region_poses[4]):
             failed_constr.append(("BasketNearLocClear", False))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        last_pose = 'ROBOT_INIT_POSE'
        if (self.state.robot_region != 3):
            ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_3_POSE_0 REGION3 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            last_pose = 'ROBOT_REGION_3_POSE_0'

        ll_plan_str.append('{0}: MOVETO BAXTER {1} BASKET_GRASP_BEGIN_0 \n'.format(act_num, last_pose))
        act_num += 1
        ll_plan_str.append('{0}: BASKET_GRASP BAXTER BASKET BASKET_FAR_TARGET BASKET_GRASP_BEGIN_0 BG_EE_LEFT_0 BG_EE_RIGHT_0 BASKET_GRASP_END_0 \n'.format(act_num))
        act_num += 1

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        self.ll_solver._backtrack_solve(plan, callback=None, amax=len(plan.actions)-2)

        for action in plan.actions:
            self.execute_plan(plan, action.active_timesteps)

        # self.predict_basket_location_from_wrist(plan)

        act_num = 0
        ll_plan_str = []

        ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE BASKET_GRASP_BEGIN_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BASKET_GRASP BAXTER BASKET BASKET_FAR_TARGET BASKET_GRASP_BEGIN_0 BG_EE_LEFT_0 BG_EE_RIGHT_0 BASKET_GRASP_END_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: ROTATE_HOLDING_BASKET BAXTER BASKET BASKET_GRASP_END_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVEHOLDING_BASKET BAXTER ROBOT_REGION_1_POSE_0 BASKET_PUTDOWN_BEGIN_0 BASKET \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BASKET_PUTDOWN BAXTER BASKET BASKET_NEAR_TARGET BASKET_PUTDOWN_BEGIN_0 BP_EE_LEFT_0 BP_EE_RIGHT_0 BASKET_PUTDOWN_END_0 \n'.format(act_num))

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        self.ll_solver.backtrack_solve(plan, callback=None)

        for action in plan.actions:
            self.execute_plan(plan, action.active_timesteps)

        act_num = 0
        ll_plan_str = []

        return []

    def move_basket_from_washer(self):
        self.predict_cloth_locations()
        self.predict_basket_location()

        failed_constr = []
        if len(self.state.region_poses[5]):
             failed_constr.append(("BasketFarLocClear", False))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        last_pose = 'ROBOT_INIT_POSE'
        if (self.state.robot_region != 3):
            ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            last_pose = 'ROBOT_REGION_1_POSE_0'

        ll_plan_str.append('{0}: MOVETO BAXTER {1} BASKET_GRASP_BEGIN_0 \n'.format(act_num, last_pose))
        act_num += 1
        ll_plan_str.append('{0}: BASKET_GRASP BAXTER BASKET BASKET_FAR_TARGET BASKET_GRASP_BEGIN_0 BG_EE_LEFT_0 BG_EE_RIGHT_0 BASKET_GRASP_END_0 \n'.format(act_num))
        act_num += 1

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        self.ll_solver._backtrack_solve(plan, callback=None, amax=len(plan.actions)-2)

        for action in plan.actions:
            self.execute_plan(plan, action.active_timesteps)

        # self.predict_basket_location_from_wrist(plan)
        
        act_num = 0
        ll_plan_str = []

        ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE BASKET_GRASP_BEGIN_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BASKET_GRASP BAXTER BASKET BASKET_FAR_TARGET BASKET_GRASP_BEGIN_0 BG_EE_LEFT_0 BG_EE_RIGHT_0 BASKET_GRASP_END_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: ROTATE_HOLDING_BASKET BAXTER BASKET BASKET_GRASP_END_0 ROBOT_REGION_3_POSE_0 REGION3 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: MOVEHOLDING_BASKET BAXTER ROBOT_REGION_3_POSE_0 BASKET_PUTDOWN_BEGIN_0 BASKET \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: BASKET_PUTDOWN BAXTER BASKET BASKET_NEAR_TARGET BASKET_PUTDOWN_BEGIN_0 BP_EE_LEFT_0 BP_EE_RIGHT_0 BASKET_PUTDOWN_END_0 \n'.format(act_num))

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        self.ll_solver.backtrack_solve(plan, callback=None)

        for action in plan.actions:
            self.execute_plan(plan, action.active_timesteps)

        return []

    def open_washer(self):
        failed_constr = []

        act_num = 0
        ll_plan_str = []

        last_pose = 'ROBOT_INIT_POSE'
        if (self.state.robot_region != 3):
            ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            last_pose = 'ROBOT_REGION_1_POSE_0'

        ll_plan_str.append('{0}: MOVETO BAXTER {1} CLOSE_DOOR_SCAN_POSE \n'.format(act_num, last_pose))

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        self.ll_solver.backtrack_solve(plan, callback=None)

        for action in plan.actions:
            self.execute_plan(plan, action.active_timesteps)

        # self.predict_open_door()
        if self.state.washer_door < -np.pi/6:
             failed_constr.append(("WasherDoorOpen", True))

        if len(failed_constr):
            return failed_constr
        
        act_num = 0
        ll_plan_str = []

        ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE OPEN_DOOR_BEGIN_0 \n'.format(act_num))
        act_num += 1
        ll_plan_str.append('{0}: OPEN_DOOR BAXTER WASHER OPEN_DOOR_BEGIN_0 OPEN_DOOR_EE_APPROACH_0 OPEN_DOOR_EE_RETREAT_0 OPEN_DOOR_END_0 WASHER_CLOSE_POSE_0 WASHER_OPEN_POSE_0 \n'.format(act_num))
        act_num += 1
        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        self.ll_solver.backtrack_solve(plan, callback=None)

        for action in plan.actions:
            self.execute_plan(plan, action.active_timesteps)

        self.state.washer_door = -np.pi/2

        return []

    def close_washer(self):
        failed_constr = []
        if self.state.washer_door > -np.pi/6:
             failed_constr.append(("WasherDoorOpen", False))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        last_pose = 'ROBOT_INIT_POSE'
        if (self.state.robot_region != 3):
            ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            last_pose = 'ROBOT_REGION_1_POSE_0'

        ll_plan_str.append('{0}: MOVETO BAXTER {1} OPEN_DOOR_SCAN_POSE \n'.format(act_num, last_pose))

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        self.ll_solver.backtrack_solve(plan, callback=None)

        for action in plan.actions:
            self.execute_plan(plan, action.active_timesteps)

        # self.predict_open_door()
        if self.state.washer_door < -np.pi/6:
             failed_constr.append(("WasherDoorOpen", True))

        if len(failed_constr):
            return failed_constr
        
        act_num = 0
        ll_plan_str = []

        ll_plan_str.append('{0}: MOVETO BAXTER {1} CLOSE_DOOR_BEGIN_0 \n'.format(act_num, last_pose))
        act_num += 1
        ll_plan_str.append('{0}: CLOSE_DOOR BAXTER WASHER CLOSE_DOOR_BEGIN_0 CLOSE_DOOR_EE_APPROACH_0 CLOSE_DOOR_EE_RETREAT_0 CLOSE_DOOR_END_0 WASHER_OPEN_POSE_0 WASHER_CLOSE_POSE_0 \n'.format(act_num))
        act_num += 1
        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        self.ll_solver.backtrack_solve(plan, callback=None)

        for action in plan.actions:
            self.execute_plan(plan, action.active_timesteps)

        self.state.washer_door = 0

        return []

    def load_washer_from_basket(self):
        self.predict_cloth_locations()
        self.predict_basket_location()

        failed_constr = []
        if not len(self.state.region_poses[4]):
             failed_constr.append(("ClothInBasket", False))

        if self.state.washer_door > -np.pi/2:
            failed_constr.append(("WasherDoorOpen", False))

        if np.any(np.abs(self.state.basket_pose - utils.basket_near_pos) > 0.1):
            failed_constr.append(("BasketNearWasher", True))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        while len(self.state.region_poses[4]):
            last_pose = 'ROBOT_INIT_POSE'
            if (self.state.robot_region != 1):
                ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                last_pose = 'ROBOT_REGION_1_POSE_0'

            ll_plan_str.append('{0}: MOVETO BAXTER {1} CLOTH_GRASP_BEGIN_0 \n'.format(act_num, last_pose))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_WASHER BAXTER WASHER WASHER_POSE_0 CLOTH0 CLOTH_TARGET_END_0 CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1
            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': 5})
            self.ll_solver.backtrack_solve(plan, callback=None)

            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

            act_num = 0
            ll_plan_str = []
            self.predict_cloth_locations()

        return []

    def load_washer_from_region_1(self):
        self.predict_cloth_locations()
        self.predict_basket_location()

        failed_constr = []
        if not len(self.state.region_poses[0]):
             failed_constr.append(("ClothInRegion1", False))

        if self.state.washer_door > -np.pi/2:
            failed_constr.append(("WasherDoorOpen", False))

        if np.any(np.abs(self.state.basket_pose - utils.basket_near_pos) > 0.05):
            failed_constr.append(("BasketNearWasher", True))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        while len(self.state.region_poses[0]):
            last_pose = 'ROBOT_INIT_POSE'
            if (self.state.robot_region != 1):
                ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                last_pose = 'ROBOT_REGION_1_POSE_0'

            ll_plan_str.append('{0}: MOVETO BAXTER {1} CLOTH_GRASP_BEGIN_0 \n'.format(act_num, last_pose))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_WASHER BAXTER WASHER WASHER_POSE_0 CLOTH0 CLOTH_TARGET_END_0 CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1
            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': 1})
            self.ll_solver.backtrack_solve(plan, callback=None)

            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

            act_num = 0
            ll_plan_str = []
            self.predict_cloth_locations()

        return []

    def clear_basket_near_loc(self):
        self.predict_cloth_locations()
        self.predict_basket_location()

        failed_constr = []
        if not len(self.state.region_poses[4]):
             failed_constr.append(("BasketNearLocClear", True))

        if self.state.washer_door > -np.pi/2:
            failed_constr.append(("WasherDoorOpen", False))

        if np.all(np.abs(self.state.basket_pose - utils.basket_near_pos) < 0.15):
            failed_constr.append(("BasketInNearLoc", True))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        while len(self.state.region_poses[4]):
            last_pose = 'ROBOT_INIT_POSE'
            if (self.state.robot_region != 1):
                ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
                act_num += 1
                last_pose = 'ROBOT_REGION_1_POSE_0'

            ll_plan_str.append('{0}: MOVETO BAXTER {1} CLOTH_GRASP_BEGIN_0 \n'.format(act_num, last_pose))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_WASHER BAXTER WASHER WASHER_POSE_0 CLOTH0 CLOTH_TARGET_END_0 CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1
            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': 4})
            self.ll_solver.backtrack_solve(plan, callback=None)

            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

            act_num = 0
            ll_plan_str = []
            self.predict_cloth_locations()

        return []

    def clear_basket_far_loc(self):
        self.predict_cloth_locations()
        self.predict_basket_location()

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

        while len(self.state.region_poses[5]):
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
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_REGION_1_POSE_0 CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_FAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1

            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, {'cloth0': 5})
            self.ll_solver.backtrack_solve(plan, callback=None)

            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

            act_num = 0
            ll_plan_str = []
            self.predict_cloth_locations()

        return []

    def unload_washer_into_basket(self):
        self.predict_cloth_locations()
        self.predict_basket_location()

        failed_constr = []
        if self.state.washer_door > -np.pi/2:
            failed_constr.append(("WasherDoorOpen", False))

        if np.any(np.abs(self.state.basket_pose - utils.basket_near_pos)[:2] > 0.1):
            failed_constr.append(("BasketNearWasher", True))

        if len(failed_constr):
            return failed_constr

        act_num = 0
        ll_plan_str = []

        last_pose = 'ROBOT_INIT_POSE'
        if (self.state.robot_region != 1):
            ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            last_pose = 'ROBOT_REGION_1_POSE_0'

        ll_plan_str.append('{0}: MOVETO BAXTER {1} WASHER_SCAN_POSE \n'.format(act_num, last_pose))
        act_num += 1
        for action in plan.actions:
            self.execute_plan(plan, action.active_timesteps)
        self.predict_cloth_washer_locations()

        while len(self.state.washer_cloth_poses):
            ll_plan_str.append('{0}: MOVETO BAXTER WASHER_SCAN_POSE CLOTH_GRASP_BEGIN_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: TAKE_OUT_OF_WASHER BAXTER WASHER WASHER_POSE_0 CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CG_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_0 WASHER_SCAN_POSE \n'.format(act_num))
            act_num += 1
            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, in_washer=True)
            self.ll_solver.backtrack_solve(plan, callback=None)

            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

            act_num = 0
            ll_plan_str = []
            self.predict_cloth_washer_locations()

        return []

    def reset_laundry(self):
        '''
        Used to have Baxter reset its environment to a good initial configuration.
        '''
        self.move_basket_from_washer()

        self.predict_cloth_locations()
        self.predict_basket_location()

        if self.state.washer_door < -np.pi/6:
            self.open_washer()

        act_num = 0
        ll_plan_str = []

        last_pose = 'ROBOT_INIT_POSE'
        if (self.state.robot_region != 1):
            ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_INIT_POSE ROBOT_REGION_{1}_POSE_0 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE BAXTER ROBOT_REGION_{1}_POSE_0 ROBOT_REGION_1_POSE_0 REGION1 \n'.format(act_num, self.state.robot_region))
            act_num += 1
            last_pose = 'ROBOT_REGION_1_POSE_0'

        ll_plan_str.append('{0}: MOVETO BAXTER {1} WASHER_SCAN_POSE \n'.format(act_num, last_pose))
        act_num += 1
        for action in plan.actions:
            self.execute_plan(plan, action.active_timesteps)
        self.predict_cloth_washer_locations()

        while len(self.state.washer_cloth_poses):
            next_region = np.random.choice([1, 2, 3, 4])
            ll_plan_str.append('{0}: MOVETO BAXTER WASHER_SCAN_POSE CLOTH_GRASP_BEGIN_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: TAKE_OUT_OF_WASHER BAXTER WASHER WASHER_POSE_0 CLOTH0 CLOTH_TARGET_BEGIN_0 CLOTH_GRASP_BEGIN_0 CP_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: ROTATE_HOLDING_CLOTH BAXTER CLOTH0 CLOTH_GRASP_END_0 ROBOT_REGION_{1}_POSE_0'.format(act_num, next_region))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_REGION_{1}_POSE_0 CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_num, next_region))
            act_num += 1
            ll_plan_str.append('{0}: CLOTH_PUTDOWN BAXTER CLOTH0 CLOTH_TARGET_END_0 CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_0 WASHER_SCAN_POSE \n'.format(act_num))
            act_num += 1
            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, in_washer=True,, reset=True)
            self.ll_solver.backtrack_solve(plan, callback=None)

            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

            act_num = 0
            ll_plan_str = []
            self.predict_cloth_washer_locations()

        self.close_washer()

        return []
