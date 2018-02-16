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
        self.cloth_in_basket = False
        self.cloth_in_washer = False
        self.basket_near_washer = False
        self.washer_door = -np.pi/2
        self.near_loc_clear = True
        self.far_loc_clear = True

        # For constructing high level plans
        self.prob_domain = "(:domain laundry_domain)\n"
        self.objects = "(:objects cloth washer basket)\n"
        self.goal = "(:goal (ClothInWasher cloth washer))\n"

    def reset_region_poses(self):
        self.region_poses = [[], [], [], []]

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

        regions_occupied = 0
        # for i in range(len(self.region_poses)):
        #     r = self.region_poses[i]
        #     if len(r):
        #         state_str += "(ClothInRegion{0} cloth)\n".format(i+1)
        #         regions_occupied += 1

        if self.cloth_in_basket:
            state_str += "(ClothInBasket cloth basket)\n"
        # if not regions_occupied:
        #     state_str += "(ClothInWasher cloth washer)\n"
        if self.basket_near_washer:
            state_str += "(BasketNearWasher basket washer)\n"
        if self.near_loc_clear:
            state_str += "(BasketNearLocClear basket cloth)\n"
        if self.far_loc_clear:
            state_str += "(BasketFarLocClear basket cloth)\n"
        if self.washer_door == -np.pi/2:
            state_str += "(WasherDoorOpen washer)\n"

        state_str += ")\n"
        return state_str


class LaundryEnvironmentMonitor(object):
    def __init__(self):
        self.state = HLLaundryState()

        with open(DOMAIN_FILE, 'r+') as f:
            self.abs_domain = f.read()
        self.hl_solver = FFSolver(abs_domain=self.abs_domain)
        self.ll_solver = RobotLLSolver()
        # self.cloth_predictor = ClothGridPredict()
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
        self.predict_cloth_locations()
        # self.predict_basket_location()
        hl_plan = self.solve_hl_prob()

        import ipdb; ipdb.set_trace()

        last_basket_pose = []
        last_basket_rot = []
        last_washer_door = []

        for action in hl_plan:
            ll_plan_str, cloth_to_region = self.hl_to_ll([action])
            plan = self.plan_from_str(ll_plan_str)
            self.update_plan(plan, cloth_to_region)

            if len(last_washer_door):
                plan.params['washer'].door[:, 0] = last_washer_door[:]
            if len(last_basket_pose):
                plan.params['basket'].pose[:, 0] = last_basket_pose[:]
            if len(last_basket_rot):
                plan.params['basket'].rotation[:, 0] = last_basket_rot[:]

            viewer = None # OpenRAVEViewer.create_viewer(plan.env)
            callback = lambda x: viewer

            self.ll_solver.backtrack_solve(plan, callback=callback)

            import ipdb; ipdb.set_trace()

            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)
            # TODO: Update high level state here & check if need to replace hl_plan
            import ipdb; ipdb.set_trace()
            last_washer_door = plan.params['washer'].door[:, -1]
            last_basket_pose = plan.params['basket'].pose[:, -1]
            last_basket_rot = plan.params['basket'].rotation[:, -1]

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
            elif cur_action.name == "basket_putdown_with_cloth":
                self.state.basket_near_washer = cur_action.params[2].name == "basket_near_target"

            if cur_action.name == "center_over_basket":
                pass
            elif cur_action.name == "center_over_cloth":
                pass
            elif cur_action.name == "center_over_washer_handle":
                pass
            elif cur_action.name.startswith("rotate"):
                old_region = self.state.robot_region
                self.state.robot_region = int(cur_action.params[-1].name[-1]) # TODO: Make this more generalized
                if self.state.robot_region != old_region:
                    self.rotate_control.rotate_to_region(self.state.robot_region)
                # TODO: Add rotation integration here
            else:
                self.traj_control.execute_plan(plan, active_ts=cur_action.active_timesteps)
            current_ts = cur_action.active_timesteps[1]

    def predict_cloth_locations(self):
        self.state.reset_region_poses()
        locs = self.cloth_predictor.predict()
        for loc in locs:
            self.state.region_poses[loc[1]-1].append(loc[0])

    def predict_basket_location(self):
        self.state.basket_pose = self.basket_predictor.predict()

    def predict_open_door(self):
        self.state.washer_door = self.door_predictor.predict()

    def predict_basket_location_from_wrist(self):
        self.state.basket_pose = self.basket_wrist_predictor.predict_basket()

    def predict_cloth_washer_locations(self):
        self.state.washer_cloth_poses = self.cloth_predictor.predict_washer()

    def update_plan(self, plan, cloth_to_region={}, in_washer=False):
        plan.params['basket'].pose[:2, 0] = self.state.basket_pose[:2]
        plan.params['basket'].rotation[0, 0] = self.state.basket_pose[2]

        left_joint_dict = self.left_arm.joint_angles()
        left_joints = [left_joint_dict[joint] for joint in const.left_joints]
        left_grip = const.GRIPPER_OPEN_VALUE if self.left_grip.position() > 50 else const.GRIPPER_CLOSE_VALUE
        right_joint_dict = self.right_arm.joint_angles()
        right_joints = [right_joint_dict[joint] for joint in const.right_joints]
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

        self.state.cloth_in_basket = True
        self.state.cloth_in_washer = True

        poses = copy.deepcopy(self.state.region_poses)
        for cloth in cloth_to_region:
            next_pose = poses[cloth_to_region[cloth]-1].pop()
            plan.params['cloth{0}'.format(cloth)].pose[:2, 0] = next_pose
            plan.params['cloth_target_begin_{0}'.format(cloth)].value[:, 0] = plan.params['cloth{0}'.format(cloth)].pose[:, 0]
            self.state.cloth_in_washer = False
            if np.any(np.abs(next_pose - self.state.basket_pose[:2] > 0.15)):
                self.state.cloth_in_basket = False

        if len(self.state.washer_cloth_poses) and in_washer:
            plan.params['cloth0'].pose[:, 0] = self.state.washer_cloth_poses[0]
            plan.params['cloth_target_begin_0'].value[:, 0] = plan.params['cloth0'].pose[:, 0]
            self.state.cloth_in_washer = True

    def plan_from_str(self, ll_plan_str):
        '''Convert a plan string (low level) into a plan object.'''
        num_cloth = sum([len(poses) for poses in self.state.region_poses])
        domain_fname = '../domains/laundry_domain/laundry.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = FFSolver(d_c)
        p_c = main.parse_file_to_dict('../domains/laundry_domain/laundry_probs/baxter_laundry_{0}.prob'.format(num_cloth))
        problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        return hls.get_plan(ll_plan_str, domain, problem)

    def load_basket_from_region_1(self):
        self.predict_cloth_locations()
        # self.predict_basket_location()

        failed_constr = []
        if not len(self.state.region_poses[0]):
             failed_constr.append(("ClothInRegion1", False))

        # if not (self.state.basket_pose == utils.basket_far_pos):
        #     failed_constr.append(("BasketNearWasher", True))

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


            if not len(self.state.region_poses[0]):
                return []

            # if not (self.state.basket_pose == utils.basket_far_pos):
            #     return [("BasketNearWasher", True)]


    def load_basket_from_region_2(self):
        self.predict_cloth_locations()
        # self.predict_basket_location()

        failed_constr = []
        if not len(self.state.region_poses[1]):
             failed_constr.append(("ClothInRegion2", False))

        # if not (self.state.basket_pose == utils.basket_far_pos):
        #     failed_constr.append(("BasketNearWasher", True))

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
            ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_FAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_numi))
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


            if not len(self.state.region_poses[1]):
                return []

            # if not (self.state.basket_pose == utils.basket_far_pos):
            #     return [("BasketNearWasher", True)]

    def load_basket_from_region_3(self):
        self.predict_cloth_locations()
        # self.predict_basket_location()

        failed_constr = []
        if not len(self.state.region_poses[2]):
             failed_constr.append(("ClothInRegion3", False))

        # if not (self.state.basket_pose == utils.basket_far_pos):
        #     failed_constr.append(("BasketNearWasher", True))

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
            # self.predict_basket_location()


            if not len(self.state.region_poses[2]):
                return []

            # if not (self.state.basket_pose == utils.basket_far_pos):
            #     return [("BasketNearWasher", True)]

    def load_basket_from_region_4(self):
        self.predict_cloth_locations()
        # self.predict_basket_location()

        failed_constr = []
        if not len(self.state.region_poses[3]):
             failed_constr.append(("ClothInRegion4", False))

        # if not (self.state.basket_pose == utils.basket_far_pos):
        #     failed_constr.append(("BasketNearWasher", True))

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
            ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH0 BASKET CLOTH_TARGET_END_0 BASKET_FAR_TARGET CLOTH_PUTDOWN_BEGIN_0 CP_EE_0 CLOTH_PUTDOWN_END_0 \n'.format(act_numi))
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


            if not len(self.state.region_poses[3]):
                return []

            # if not (self.state.basket_pose == utils.basket_far_pos):
            #     return [("BasketNearWasher", True)]

    def move_basket_to_washer(self):
        self.predict_cloth_locations()
        self.predict_basket_location()

        failed_constr = []
        if len(self.state.region_poses[4]):
             failed_constr.append(("BasketNearLocClear", False))

        if not (self.state.basket_pose == utils.basket_far_pos):
            failed_constr.append(("BasketNearWasher", True))

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

        ll_plan_str.append('{0}: MOVETO BAXTER {1} ROBOT_REGION_3_SCAN_POSE \n'.format(act_num, last_pose))

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        self.ll_solver.backtrack_solve(plan, callback=None)

        for action in plan.actions:
            self.execute_plan(plan, action.active_timesteps)

        self.predict_basket_location_from_wrist()

        act_num = 0
        ll_plan_str = []

        ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_3_SCAN_POSE BASKET_GRASP_BEGIN_0 \n'.format(act_num))
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
        self.predict_basket_location()

        import ipdb; ipdb.set_trace()

        if not (self.state.basket_pose == utils.basket_far_pos):
            return [("BasketNearWasher", True)]

    def move_basket_from_washer(self):
        self.predict_cloth_locations()
        # self.predict_basket_location()

        failed_constr = []
        if len(self.state.region_poses[5]):
             failed_constr.append(("BasketFarLocClear", False))

        # if not (self.state.basket_pose == utils.basket_far_pos):
        #     failed_constr.append(("BasketNearWasher", True))

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

        ll_plan_str.append('{0}: MOVETO BAXTER {1} ROBOT_REGION_1_SCAN_POSE \n'.format(act_num, last_pose))

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        self.ll_solver.backtrack_solve(plan, callback=None)

        for action in plan.actions:
            self.execute_plan(plan, action.active_timesteps)

        self.predict_basket_location_from_wrist()
        
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

        act_num = 0
        ll_plan_str = []
        # self.predict_basket_location()

        # if not (self.state.basket_pose == utils.basket_near_pos):
        #     return [("BasketNearWasher", True)]

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

        ll_plan_str.append('{0}: MOVETO BAXTER {1} ROBOT_OPEN_DOOR_SCAN_POSE \n'.format(act_num, last_pose))

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        self.ll_solver.backtrack_solve(plan, callback=None)

        for action in plan.actions:
            self.execute_plan(plan, action.active_timesteps)

        self.predict_open_door()
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

        ll_plan_str.append('{0}: MOVETO BAXTER {1} ROBOT_CLOSE_DOOR_SCAN_POSE \n'.format(act_num, last_pose))

        plan = self.plan_from_str(ll_plan_str)
        self.update_plan(plan, {})
        self.ll_solver.backtrack_solve(plan, callback=None)

        for action in plan.actions:
            self.execute_plan(plan, action.active_timesteps)

        self.predict_open_door()
        if self.state.washer_door > -np.pi/6:
             failed_constr.append(("WasherDoorOpen", FALSE))

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

    def load_washer_from_basket(self):
        self.predict_cloth_locations()
        # self.predict_basket_location()

        failed_constr = []
        if not len(self.state.region_poses[4]):
             failed_constr.append(("ClothInBasket", False))

        if self.state.washer_door > -np.pi/2:
            failed_constr.append(("WasherDoorOpen", False))

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
            # self.predict_basket_location()

            # if not (self.state.basket_pose == utils.basket_near_pos):
            #     return [("BasketNearWasher", True)]

    def load_washer_from_region1(self):
        self.predict_cloth_locations()
        # self.predict_basket_location()

        failed_constr = []
        if not len(self.state.region_poses[0]):
             failed_constr.append(("ClothInRegion1", False))

        if self.state.washer_door > -np.pi/2:
            failed_constr.append(("WasherDoorOpen", False))

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
            self.update_plan(plan, {'cloth0': 1})
            self.ll_solver.backtrack_solve(plan, callback=None)

            for action in plan.actions:
                self.execute_plan(plan, action.active_timesteps)

            act_num = 0
            ll_plan_str = []
            self.predict_cloth_locations()
            # self.predict_basket_location()

            # if not (self.state.basket_pose == utils.basket_near_pos):
            #     return [("BasketNearWasher", True)]

    def unload_washer_into_basket(self):
        self.predict_cloth_locations()
        # self.predict_basket_location()

        failed_constr = []
        if self.state.washer_door > -np.pi/2:
            failed_constr.append(("WasherDoorOpen", False))

        # if not (self.state.basket_pose == utils.basket_near_pos):
        #     failed_constr.append(("BasketNearWasher", False))

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
            ll_plan_str.append('{0}: TAKE_OUT_OF_WASHER BAXTER WASHER WASHER_POSE_0 CLOTH0 CLOTH_TARGET_END_0 CLOTH_GRASP_BEGIN_0 CP_EE_0 CLOTH_GRASP_END_0 \n'.format(act_num))
            act_num += 1
            ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_0 CLOTH_PUTDOWN_BEGIN_0 CLOTH0 \n'.format(act_numn))
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
            # self.predict_basket_location()

            # if not (self.state.basket_pose == utils.basket_near_pos):
            #     return [("BasketNearWasher", True)]


    def hl_to_ll(self, hl_plan):
        '''Parses a high level plan into a sequence of low level actions.'''
        ll_plan_str = []
        act_num = 0
        cur_cloth_n = 0
        last_pose = 'ROBOT_INIT_POSE'
        cloth_to_region = {}
        # TODO: Fill in eval functions for basket setdown region

        i = 0
        for action in hl_plan:
            act_type = action.split()[1].lower()
            if act_type == 'load_basket_from_region_1':
                init_i = i
                ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_1_POSE_{2} REGION1 \n'.format(act_num, last_pose, i))
                act_num += 1
                while i < len(self.state.region_poses[0]) + init_i:
                    ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_1_POSE_{1} CLOTH_GRASP_BEGIN_{2} \n'.format(act_num, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5} \n'.format(act_num, cur_cloth_n, cur_cloth_n, i, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH{3} \n'.format(act_num, i, i, cur_cloth_n))
                    act_num += 1
                    ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH{1} BASKET CLOTH_TARGET_END_{2} BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5} \n'.format(act_num, cur_cloth_n, cur_cloth_n, i, i, i))
                    act_num += 1
                    i += 1
                    cloth_to_region[cur_cloth_n] = 1
                    cur_cloth_n += 1
                last_pose = 'CLOTH_PUTDOWN_END_{0}'.format(i-1)


            elif act_type == 'load_basket_from_region_2':
                init_i = i
                while i < len(self.state.region_poses[1]) + init_i:
                    ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_2_POSE_{2} REGION2 \n'.format(act_num, last_pose, i))
                    act_num += 1
                    ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_2_POSE_{1} CLOTH_GRASP_BEGIN_{2} \n'.format(act_num, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5} \n'.format(act_num, cur_cloth_n, cur_cloth_n, i, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: ROTATE_HOLDING_CLOTH BAXTER CLOTH{1} CLOTH_GRASP_END_{2} ROBOT_REGION_3_POSE_{3} REGION3 \n'.format(act_num, cur_cloth_n, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_REGION_3_POSE_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH{3} \n'.format(act_num, i, i, cur_cloth_n))
                    act_num += 1
                    ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH{1} BASKET CLOTH_TARGET_END_{2} BASKET_FAR_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5} \n'.format(act_num, cur_cloth_n, cur_cloth_n, i, i, i))
                    act_num += 1
                    i += 1
                    cloth_to_region[cur_cloth_n] = 2
                    cur_cloth_n += 1
                last_pose = 'CLOTH_PUTDOWN_END_{0}'.format(i-1)

            # The basket, when not near the washer, is in region 3
            elif act_type == 'load_basket_from_region_3':
                init_i = i
                ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_3_POSE_{2} REGION3 \n'.format(act_num, last_pose, i))
                act_num += 1
                while i < len(self.state.region_poses[2]) + init_i:
                    ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_3_POSE_{1} CLOTH_GRASP_BEGIN_{2} \n'.format(act_num, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5} \n'.format(act_num, cur_cloth_n, cur_cloth_n, i, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH{3} \n'.format(act_num, i, i, cur_cloth_n))
                    act_num += 1
                    ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH{1} BASKET CLOTH_TARGET_END_{2} BASKET_FAR_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5} \n'.format(act_num, cur_cloth_n, cur_cloth_n, i, i, i))
                    act_num += 1
                    i += 1
                    cloth_to_region[cur_cloth_n] = 3
                    cur_cloth_n += 1
                last_pose = 'CLOTH_PUTDOWN_END_{0}'.format(i-1)

            # TODO: Add right handed grasp functionality
            elif act_type == 'load_basket_from_region_4':
                init_i = i
                while i < len(self.state.region_poses[3]) + init_i:
                    ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_4_POSE_{2} REGION4 \n'.format(act_num, last_pose, i))
                    act_num += 1
                    ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_4_POSE_{1} CLOTH_GRASP_BEGIN_{2} \n'.format(act_num, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5} \n'.format(act_num, cur_cloth_n, cur_cloth_n, i, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: ROTATE_HOLDING_CLOTH BAXTER CLOTH{1} CLOTH_GRASP_END_{2} ROBOT_REGION_3_POSE_{3} REGION3 \n'.format(act_num, cur_cloth_n, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_REGION_3_POSE_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH{3} \n'.format(act_num, i, i, cur_cloth_n))
                    act_num += 1
                    ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH{1} BASKET CLOTH_TARGET_END_{2} BASKET_FAR_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5} \n'.format(act_num, cur_cloth_n, cur_cloth_n, i, i, i))
                    act_num += 1
                    i += 1
                    cloth_to_region[cur_cloth_n] = 4
                    cur_cloth_n += 1
                last_pose = 'CLOTH_PUTDOWN_END_{0}'.format(i-1)


            elif act_type == 'move_basket_to_washer':
                ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_3_POSE_{2} REGION3 \n'.format(act_num, last_pose, i))
                act_num += 1
                ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_3_POSE_{1} BASKET_GRASP_BEGIN_{2} \n'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: BASKET_GRASP BAXTER BASKET BASKET_FAR_TARGET BASKET_GRASP_BEGIN_{1} BG_EE_LEFT_{2} BG_EE_RIGHT_{3} BASKET_GRASP_END_{4} \n'.format(act_num, i, i, i, i))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE_HOLDING_BASKET BAXTER BASKET BASKET_GRASP_END_{1} ROBOT_REGION_1_POSE_{2} REGION1 \n'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: MOVEHOLDING_BASKET BAXTER ROBOT_REGION_1_POSE_{1} BASKET_PUTDOWN_BEGIN_{2} BASKET \n'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: BASKET_PUTDOWN BAXTER BASKET BASKET_NEAR_TARGET BASKET_PUTDOWN_BEGIN_{1} BP_EE_LEFT_{2} BP_EE_RIGHT_{3} BASKET_PUTDOWN_END_{4} \n'.format(act_num, i, i, i, i))
                last_pose = 'BASKET_PUTDOWN_END_{0}'.format(i)
                i += 1


            elif act_type == 'move_basket_from_washer':
                ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_1_POSE_{2} REGION1 \n'.format(act_num, last_pose, i))
                act_num += 1
                ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_1_POSE_{1} BASKET_GRASP_BEGIN_{2} \n'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: BASKET_GRASP BAXTER BASKET BASKET_NEAR_TARGET BASKET_GRASP_BEGIN_{1} BG_EE_LEFT_{2} BG_EE_RIGHT_{3} BASKET_GRASP_END_{4} \n'.format(act_num, i, i, i, i))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE_HOLDING_BASKET BAXTER BASKET BASKET_GRASP_END_{1} ROBOT_REGION_3_POSE_{2} REGION3 \n'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: MOVEHOLDING_BASKET BAXTER ROBOT_REGION_3_POSE_{1} BASKET_PUTDOWN_BEGIN_{2} BASKET \n'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: BASKET_PUTDOWN BAXTER BASKET BASKET_FAR_TARGET BASKET_PUTDOWN_BEGIN_{1} BP_EE_LEFT_{2} BP_EE_RIGHT_{3} BASKET_PUTDOWN_END_{4} \n'.format(act_num, i, i, i, i))
                last_pose = 'BASKET_PUTDOWN_END_{0}'.format(i)
                i += 1


            elif act_type == 'open_washer':
                ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_1_POSE_{2} REGION1 \n'.format(act_num, last_pose, i))
                act_num += 1
                ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_1_POSE_{1} OPEN_DOOR_BEGIN_{2} \n'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: OPEN_DOOR BAXTER WASHER OPEN_DOOR_BEGIN_{1} OPEN_DOOR_EE_APPROACH_{2} OPEN_DOOR_EE_RETREAT_{3} OPEN_DOOR_END_{4} WASHER_CLOSE_POSE_{6} WASHER_OPEN_POSE_{5} \n'.format(act_num, i, i, i, i, i, i))
                act_num += 1
                last_pose = 'OPEN_DOOR_END_{0}'.format(i)
                i += 1


            elif act_type == 'close_washer':
                ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_1_POSE_{2} REGION1 \n'.format(act_num, last_pose, i))
                act_num += 1
                ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_1_POSE_{1} CLOSE_DOOR_BEGIN_{2} \n'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: CLOSE_DOOR BAXTER WASHER CLOSE_DOOR_BEGIN_{1} CLOSE_DOOR_EE_APPROACH_{2} CLOSE_DOOR_EE_RETREAT_{3} CLOSE_DOOR_END_{4} WASHER_OPEN_POSE_{5} WASHER_CLOSE_POSE_{6} \n'.format(act_num, i, i, i, i, i, i))
                act_num += 1
                last_pose = 'CLOSE_DOOR_END_{0}'.format(i)
                i += 1


            elif act_type == 'load_washer':
                init_i = i
                cur_cloth_n = 0
                ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_1_POSE_{2} REGION1 \n'.format(act_num, last_pose, i))
                act_num += 1
                while i < num_cloths + init_i:
                    ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_1_POSE_{1} CLOTH_GRASP_BEGIN_{2} \n'.format(act_num, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5} \n'.format(act_num, cur_cloth_n, i, i, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH{3} \n'.format(act_num, i, i, cur_cloth_n))
                    act_num += 1
                    ll_plan_str.append('{0}: PUT_INTO_WASHER BAXTER WASHER WASHER_POSE_{1} CLOTH{2} CLOTH_TARGET_END_{3} CLOTH_PUTDOWN_BEGIN_{4} CP_EE_{5} CLOTH_PUTDOWN_END_{6} \n'.format(act_num, i, cur_cloth_n, i, i, i, i))
                    act_num += 1
                    i += 1
                    cur_cloth_n += 1
                i -= 1
                ll_plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_{1} CLOSE_DOOR_BEGIN_{2} \n'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: CLOSE_DOOR BAXTER WASHER CLOSE_DOOR_BEGIN_{1} CLOSE_DOOR_EE_APPROACH_{2} CLOSE_DOOR_EE_RETREAT_{3} CLOSE_DOOR_END_{4} WASHER_OPEN_POSE_{5} WASHER_CLOSE_POSE_{6} \n'.format(act_num, i, i, i, i, i, i))
                act_num += 1
                last_pose = 'CLOSE_DOOR_END_{0}'.format(i)
                i += 1
                cur_cloth_n = 0


            elif act_type == 'unload_washer':
                init_i = i
                cur_cloth_n = 0
                ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_1_POSE_{2} REGION1 \n'.format(act_num, last_pose, i))
                act_num += 1
                while i < num_cloths + init_i:
                    ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_1_POSE_{1} CLOTH_GRASP_BEGIN_{2} \n'.format(act_num, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: TAKE_OUT_OF_WASHER BAXTER WASHER WASHER_POSE_{1} CLOTH{2} CLOTH_TARGET_END_{3} CLOTH_GRASP_BEGIN_{4} CP_EE_{5} CLOTH_GRASP_END_{6} \n'.format(act_num, i, cur_cloth_n, i, i, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH{3} \n'.format(act_num, i, i, cur_cloth_n))
                    act_num += 1
                    ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH{1} BASKET CLOTH_TARGET_END_{2} BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5} \n'.format(act_num, cur_cloth_n, i, i, i, i))
                    act_num += 1
                    i += 1
                    cur_cloth_n += 1
                i -= 1
                ll_plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_{1} CLOSE_DOOR_BEGIN_{2} \n'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: CLOSE_DOOR BAXTER WASHER CLOSE_DOOR_BEGIN_{1} CLOSE_DOOR_EE_APPROACH_{2} CLOSE_DOOR_EE_RETREAT_{3} CLOSE_DOOR_END_{4} WASHER_OPEN_POSE_{5} WASHER_CLOSE_POSE_{6} \n'.format(act_num, i, i, i, i, i, i))
                act_num += 1
                last_pose = 'CLOSE_DOOR_END_{0}'.format(i)
                i += 1


        # ll_plan_str.append('{0}: MOVETO BAXTER {1} ROBOT_END_POSE \n'.format(act_num, last_pose))

        return ll_plan_str, cloth_to_region

    def _get_action_type(self, action):
        pass
