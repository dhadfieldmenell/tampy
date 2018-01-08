import numpy as np

import rospy

from ros_interace.basket.basket_predict import BasketPredict
from ros_interace.cloth.cloth_grid_predict import ClothGridPredict


class HLLaundryState(object):
    # TODO: This class is here for temporary usage but needs generalization
    #       Ideally integreate with existing HLState class and Env Monitor
    def __init__(self):
        self.region_poses = [[], [], [], []]
        self.basket_pose = [np.nan, np.nan, np.nan]

        self.robot_region = 1
        self.cloth_in_basket = False
        self.cloth_in_washer = False
        self.basket_near_washer = False
        self.washer_door_open = False
        self.near_loc_clear = True
        self.far_loc_clear = True

    def reset_region_poses(self):
        self.region_poses = [[], [], [], []]


class LaundryEnvironmentMonitor(object):
    def __init__(self):
        self.look_for_cloths = True
        self.state = HLLaundryState()
        self.cloth_predictor = ClothGridPredict()
        self.basket_predictor = BasketPredict()

    def predict_cloth_locations(self):
        if self.look_for_cloths:
            self.state.reset_region_poses()
            locs = self.cloth_predictor.predict()
            for loc in locs:
                self.state.region_poses[loc[1]].append(loc[0])

    def predict_basket_location(self):
        self.state.basket_pose = self.basket_predictor.predict()

    def update_plan(self, plan):
        plan.params['basket'].pose[:2, 0] = self.state.basket_pose[:2]
        plan.params['basket'].rotation[2, 0] = self.state.basket_pose[2]
        cur_cloth_n = 0
        for region in self.state.region_poses:
            for pose in region:
                plan.params['cloth_{0}'.format(cur_cloth_n)].pose[:2] = pose

    def hl_to_ll(self, plan_str):
        '''Parses a high level plan into a sequence of low level actions.'''
        ll_plan_str = ''
        act_num = 0
        cur_cloth_n = 0
        last_pose = 'ROBOT_INIT_POSE'
        # TODO: Fill in eval functions for basket setdown region

        for action in plan_str:
            act_type = self._get_action_type(action)
            i = 0
            if act_type == 'load_basket_from_region_1':
                init_i = i
                ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_1_POSE_{2} REGION_1'.format(act_num, last_pose, i))
                act_num += 1
                while i < num_cloths_r1 + init_i:
                    ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_1_POSE_{1} CLOTH_GRASP_BEGIN_{2}'.format(act_num, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH_{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5}'.format(act_num, cur_cloth_n, i, i, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH_{3}'.format(act_num, i, i, cur_cloth_n))
                    act_num += 1
                    ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH_{1} BASKET CLOTH_TARGET_END_{2} BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5}'.format(act_num, cur_cloth_n, i, i, i, i))
                    act_num += 1
                    i += 1
                    cur_cloth_n += 1
                last_pose = 'CLOTH_PUTDOWN_END_{0}'.format(i-1)


            elif act_type == 'load_basket_from_region_2':
                init_i = i
                while i < num_cloths_r4 + init_i:
                    ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_2_POSE_{2} REGION_2'.format(act_num, last_pose, i))
                    act_num += 1
                    ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_2_POSE_{1} CLOTH_GRASP_BEGIN_{2}'.format(act_num, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH_{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5}'.format(act_num, cur_cloth_n, i, i, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: ROTATE_HOLDING_CLOTH BAXTER CLOTH CLOTH_GRASP_END_{1} ROBOT_REGION_2_POSE_{2} REGION_2'.format(act_num, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_REGION_2_POSE_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH_{3}'.format(act_num, i, i, cur_cloth_n))
                    act_num += 1
                    ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH_{1} BASKET CLOTH_TARGET_END_{2} BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5}'.format(act_num, cur_cloth_n, i, i, i, i))
                    act_num += 1
                    i += 1
                    cur_cloth_n += 1
                last_pose = 'CLOTH_PUTDOWN_END_{0}'.format(i-1)

            # The basket, when not near the washer, is in region 3
            elif act_type == 'load_basket_from_region_3':
                init_i = i
                ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_3_POSE_{2} REGION_3'.format(act_num, last_pose, i))
                act_num += 1
                while i < num_cloths_r3 + init_i:
                    ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_3_POSE_{1} CLOTH_GRASP_BEGIN_{2}'.format(act_num, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH_{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5}'.format(act_num, cur_cloth_n, i, i, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_REGION_2_POSE_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH_{3}'.format(act_num, i, i, cur_cloth_n))
                    act_num += 1
                    ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH_{1} BASKET CLOTH_TARGET_END_{2} BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5}'.format(act_num, cur_cloth_n, i, i, i, i))
                    act_num += 1
                    i += 1
                    cur_cloth_n += 1
                last_pose = 'CLOTH_PUTDOWN_END_{0}'.format(i-1)

            # TODO: Add right handed grasp functionality
            elif act_type == 'load_basket_from_region_4':
                init_i = i
                while i < num_cloths_r4 + init_i:
                    ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_4_POSE_{2} REGION_4'.format(act_num, last_pose, i))
                    act_num += 1
                    ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_4_POSE_{1} CLOTH_GRASP_BEGIN_{2}'.format(act_num, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH_{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5}'.format(act_num, cur_cloth_n, i, i, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: ROTATE_HOLDING_CLOTH BAXTER CLOTH CLOTH_GRASP_END_{1} ROBOT_REGION_2_POSE_{2} REGION_2'.format(act_num, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER ROBOT_REGION_2_POSE_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH_{3}'.format(act_num, i, i, cur_cloth_n))
                    act_num += 1
                    ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH_{1} BASKET CLOTH_TARGET_END_{2} BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5}'.format(act_num, cur_cloth_n, i, i, i, i))
                    act_num += 1
                    i += 1
                    cur_cloth_n += 1
                last_pose = 'CLOTH_PUTDOWN_END_{0}'.format(i-1)


            elif act_type == 'move_basket_to_washer':
                ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_3_POSE_{2} REGION_3'.format(act_num, last_pose, i))
                act_num += 1
                ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_3_POSE_{1} BASKET_GRASP_BEGIN_{2}'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: BASKET_GRASP BAXTER BASKET BASKET_FAR_TARGET BASKET_GRASP_BEGIN_{1} BG_EE_LEFT_{2} BG_EE_RIGHT_{3} BASKET_GRASP_END_{4}'.format(act_num, i, i, i))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE_HOLDING_BASKET_WITH_CLOTHS BAXTER BASKET BASKET_GRASP_END_{1} ROBOT_REGION_1_POSE_{2} REGION_1'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: MOVEHOLDING_BASKET BAXTER ROBOT_REGION_1_POSE_{1} BASKET_PUTDOWN_BEGIN_{2} BASKET'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: BASKET_PUTDOWN BAXTER BASKET BASKET_NEAR_TARGET BASKET_PUTDOWN_BEGIN_{1} BP_EE_LEFT_{2} BP_EE_RIGHT_{3} BASKET_PUTDOWN_END_{4}'.format(act_num, i, i, i))
                last_pose = 'BASKET_PUTDOWN_END_{0}'.format(i)
                i += 1


            elif act_type == 'move_basket_from_washer':
                ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_1_POSE_{2} REGION_1'.format(act_num, last_pose, i))
                act_num += 1
                ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_1_POSE_{1} BASKET_GRASP_BEGIN_{2}'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: BASKET_GRASP BAXTER BASKET BASKET_NEAR_TARGET BASKET_GRASP_BEGIN_{1} BG_EE_LEFT_{2} BG_EE_RIGHT_{3} BASKET_GRASP_END_{4}'.format(act_num, i, i, i))
                act_num += 1
                ll_plan_str.append('{0}: ROTATE_HOLDING_BASKET_WITH_CLOTHS BAXTER BASKET BASKET_GRASP_END_{1} ROBOT_REGION_3_POSE_{2} REGION_3'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: MOVEHOLDING_BASKET BAXTER ROBOT_REGION_3_POSE_{1} BASKET_PUTDOWN_BEGIN_{2} BASKET'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: BASKET_PUTDOWN BAXTER BASKET BASKET_FARR_TARGET BASKET_PUTDOWN_BEGIN_{1} BP_EE_LEFT_{2} BP_EE_RIGHT_{3} BASKET_PUTDOWN_END_{4}'.format(act_num, i, i, i))
                last_pose = 'BASKET_PUTDOWN_END_{0}'.format(i)
                i += 1


            elif act_type == 'open_washer':
                ll_plan_str.append('{0}: MOVETO BAXTER {1} OPEN_DOOR_BEGIN_{2}'.format(act_num, last_pose, i))
                act_num += 1
                ll_plan_str.append('{0}: OPEN_DOOR BAXTER WASHER OPEN_DOOR_BEGIN_{1} OPEN_DOOR_EE_APPROACH_{2} OPEN_DOOR_EE_RETEREAT_{3} OPEN_DOOR_END_{4} WASHER_OPEN_POSE_{5} WASHER_CLOSE_POSE_{6}'.format(act_num, i, i, i, i, i, i))
                act_num += 1
                i += 1


            elif act_type == 'close_washer':
                ll_plan_str.append('{0}: MOVETO BAXTER {1} CLOSE_DOOR_BEGIN_{2}'.format(act_num, last_pose, i))
                act_num += 1
                ll_plan_str.append('{0}: CLOSE_DOOR BAXTER WASHER CLOSE_DOOR_BEGIN_{1} CLOSE_DOOR_EE_APPROACH_{2} CLOSE_DOOR_EE_RETEREAT_{3} CLOSE_DOOR_END_{4} WASHER_OPEN_POSE_{5} WASHER_CLOSE_POSE_{6}'.format(act_num, i, i, i, i, i, i))
                act_num += 1
                i += 1


            elif act_type == 'load_washer':
                init_i = i
                cur_cloth_n = 0
                ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_1_POSE_{2} REGION_1'.format(act_num, last_pose, i))
                act_num += 1
                while i < num_cloths + init_i:
                    ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_1_POSE_{1} CLOTH_GRASP_BEGIN_{2}'.format(act_num, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: CLOTH_GRASP BAXTER CLOTH_{1} CLOTH_TARGET_BEGIN_{2} CLOTH_GRASP_BEGIN_{3} CG_EE_{4} CLOTH_GRASP_END_{5}'.format(act_num, cur_cloth_n, i, i, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH_{3}'.format(act_num, i, i, cur_cloth_n))
                    act_num += 1
                    ll_plan_str.append('{0}: PUT_INTO_WASHER BAXTER WASHER WASHER_POSE_{1} CLOTH_{2} CLOTH_TARGET_END_{3} CLOTH_PUTDOWN_BEGIN_{4} CP_EE_{5} CLOTH_PUTDOWN_END_{6}'.format(act_num, i, cur_cloth_n, i, i, i, i))
                    act_num += 1
                    i += 1
                    cur_cloth_n += 1
                i -= 1
                ll_plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_{1} CLOSE_DOOR_BEGIN_{2}'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: CLOSE_DOOR BAXTER WASHER CLOSE_DOOR_BEGIN_{1} CLOSE_DOOR_EE_APPROACH_{2} CLOSE_DOOR_EE_RETEREAT_{3} CLOSE_DOOR_END_{4} WASHER_OPEN_POSE_{5} WASHER_CLOSE_POSE_{6}'.format(act_num, i, i, i, i, i, i))
                act_num += 1
                last_pose = 'CLOSE_DOOR_END_{0}'.format(i)
                i += 1


            elif act_type == 'unload_washer':
                init_i = i
                cur_cloth_n = 0
                ll_plan_str.append('{0}: ROTATE BAXTER {1} ROBOT_REGION_1_POSE_{2} REGION_1'.format(act_num, last_pose, i))
                act_num += 1
                while i < num_cloths + init_i:
                    ll_plan_str.append('{0}: MOVETO BAXTER ROBOT_REGION_1_POSE_{1} CLOTH_GRASP_BEGIN_{2}'.format(act_num, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: TAKE_OUT_OF_WASHER BAXTER WASHER WASHER_POSE_{1} CLOTH_{2} CLOTH_TARGET_END_{3} CLOTH_GRASP_BEGIN_{4} CP_EE_{5} CLOTH_GRASP_END_{6}'.format(act_num, i, cur_cloth_n, i, i, i, i))
                    act_num += 1
                    ll_plan_str.append('{0}: MOVEHOLDING_CLOTH BAXTER CLOTH_GRASP_END_{1} CLOTH_PUTDOWN_BEGIN_{2} CLOTH_{3}'.format(act_num, i, i, cur_cloth_n))
                    act_num += 1
                    ll_plan_str.append('{0}: PUT_INTO_BASKET BAXTER CLOTH_{1} BASKET CLOTH_TARGET_END_{2} BASKET_NEAR_TARGET CLOTH_PUTDOWN_BEGIN_{3} CP_EE_{4} CLOTH_PUTDOWN_END_{5}'.format(act_num, cur_cloth_n, i, i, i, i))
                    act_num += 1
                    i += 1
                    cur_cloth_n += 1
                i -= 1
                ll_plan_str.append('{0}: MOVETO BAXTER CLOTH_PUTDOWN_END_{1} CLOSE_DOOR_BEGIN_{2}'.format(act_num, i, i))
                act_num += 1
                ll_plan_str.append('{0}: CLOSE_DOOR BAXTER WASHER CLOSE_DOOR_BEGIN_{1} CLOSE_DOOR_EE_APPROACH_{2} CLOSE_DOOR_EE_RETEREAT_{3} CLOSE_DOOR_END_{4} WASHER_OPEN_POSE_{5} WASHER_CLOSE_POSE_{6}'.format(act_num, i, i, i, i, i, i))
                act_num += 1
                last_pose = 'CLOSE_DOOR_END_{0}'.format(i)
                i += 1


        ll_plan_str.append('{0}: MOVETO BAXTER {1} ROBOT_END_POSE'.format(act_num, last_pose))

        return ll_plan_str

    def _get_action_type(self, action):
        pass
