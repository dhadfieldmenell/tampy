# import matplotlib as mpl
# mpl.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import openravepy
import os
from threading import Thread
import time
import xml.etree.ElementTree as xml

from dm_control import render
from dm_control.mujoco import Physics
from dm_control.viewer import gui
from dm_control.viewer import renderer
from dm_control.viewer import runtime
from dm_control.viewer import user_input
from dm_control.viewer import util
from dm_control.viewer import viewer
from dm_control.viewer import views

from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.robots import Baxter
from policy_hooks.baxter.baxter_ik_controller import BaxterIKController
from policy_hooks.utils.mjc_xml_utils import *
import policy_hooks.utils.transform_utils as T


BASE_POS_XML = '../models/baxter/mujoco/baxter_mujoco_pos.xml'
BASE_VEL_XML = '../models/baxter/mujoco/baxter_vel.xml'
BASE_MOTOR_XML = '../models/baxter/mujoco/baxter_mujoco.xml'
ENV_XML = 'policy_hooks/mujoco/current_baxter_env.xml'


MUJOCO_JOINT_ORDER = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2', 'right_gripper_l_finger_joint', 'right_gripper_r_finger_joint',\
                      'left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2', 'left_gripper_l_finger_joint', 'left_gripper_r_finger_joint']


NO_CLOTH = 0
NO_FOLD = 1
ONE_FOLD = 2
TWO_FOLD = 3
WIDTH_GRASP = 4
LENGTH_GRASP = 5
TWO_GRASP = 6
HALF_WIDTH_GRASP = 7
HALF_LENGTH_GRASP = 8
TWIST_FOLD = 9

# BAXTER_GAINS = {
#     'left_s0': (700., 0.01, 25.),
#     'left_s1': (10000., 100., 100.),
#     'left_e0': (4500., 35., 1.),
#     'left_e1': (5500, 60, 2),
#     'left_w0': (1000, 30, 0.01),
#     'left_w1': (900, 0.1, 0.01),
#     'left_w2': (1000, 0.1, 0.01),
#     'left_gripper_l_finger_joint': (1000, 0.1, 0.01),
#     'left_gripper_r_finger_joint': (1000, 0.1, 0.01),

#     'right_s0': (700., 0.01, 25.),
#     'right_s1': (10000., 100., 100.),
#     'right_e0': (4500., 35., 1.),
#     'right_e1': (5500, 60, 2),
#     'right_w0': (1000, 30, 0.01),
#     'right_w1': (900, 0.1, 0.01),
#     'right_w2': (1000, 0.1, 0.01),
#     'right_gripper_l_finger_joint': (1000, 0.1, 0.01),
#     'right_gripper_r_finger_joint': (1000, 0.1, 0.01),
# }

BAXTER_GAINS = {
    'left_s0': (5000., 0.01, 2.5),
    'left_s1': (5000., 50., 50.),
    'left_e0': (4000., 15., 1.),
    'left_e1': (1500, 30, 1.),
    'left_w0': (500, 10, 0.01),
    'left_w1': (500, 0.1, 0.01),
    'left_w2': (1000, 0.1, 0.01),
    'left_gripper_l_finger_joint': (1000, 0.1, 0.01),
    'left_gripper_r_finger_joint': (1000, 0.1, 0.01),

    'right_s0': (5000., 0.01, 2.5),
    'right_s1': (5000., 50., 50.),
    'right_e0': (4000., 15., 1.),
    'right_e1': (1500, 30, 1.),
    'right_w0': (500, 10, 0.01),
    'right_w1': (500, 0.1, 0.01),
    'right_w2': (1000, 0.1, 0.01),
    'right_gripper_l_finger_joint': (1000, 0.1, 0.01),
    'right_gripper_r_finger_joint': (1000, 0.1, 0.01),
}

_MAX_FRONTBUFFER_SIZE = 2048
_CAM_WIDTH = 200
_CAM_HEIGHT = 150

GRASP_THRESHOLD = np.array([0.05, 0.05, 0.025]) # np.array([0.01, 0.01, 0.03])
MJC_TIME_DELTA = 0.002
MJC_DELTAS_PER_STEP = int(1. // MJC_TIME_DELTA)
N_CONTACT_LIMIT = 12

START_EE = [0.6, -0.5, 0.75, 0, 0, 1, 0, 0.6, 0.5, 0.75, 0, 0, 1, 0]
CTRL_MODES = ['joint_angle', 'end_effector', 'end_effector_pos', 'discrete_pos']



class BaxterMJCEnv(object):
    def __init__(self, mode='joint_angle', obs_include=[], items=[], cloth_info=None, view=False):
        assert mode in CTRL_MODES, 'Env mode must be one of {0}'.format(CTRL_MODES)
        self.ctrl_mode = mode
        self.active = True

        self.cur_time = 0.
        self.prev_time = 0.

        self.use_viewer = view
        self.use_glew = os.environ['MUJOCO_GL'] != 'osmesa'
        self.obs_include = obs_include
        self._obs_inds = {}
        self._joint_map_cache = {}
        self._ind_cache = {}
        self._cloth_present = cloth_info is not None
        if self._cloth_present:
            self.cloth_width = cloth_info['width']
            self.cloth_length = cloth_info['length']
            self.cloth_sphere_radius = cloth_info['radius']
            self.cloth_spacing = cloth_info['spacing']

        ind = 0
        if 'image' in obs_include or not len(obs_include):
            self._obs_inds['image'] = (ind, ind+_CAM_WIDTH*_CAM_HEIGHT)
            ind += _CAM_WIDTH*_CAM_HEIGHT
        if 'joints' in obs_include or not len(obs_include):
            self._obs_inds['joints'] = (ind, ind+18)
            ind += 18
        if 'end_effector' in obs_include or not len(obs_include):
            self._obs_inds['end_effector'] = (ind, ind+12)
            ind += 12
        for item, xml, info in items:
            if item in obs_include or not len(obs_include):
                self._obs_inds['image'] = (ind, ind+6)
                ind += 6


        self.ctrl_data = {}
        for joint in BAXTER_GAINS:
            self.ctrl_data[joint] = {
                'prev_err': 0.,
                'cp': 0.,
                'cd': 0.,
                'ci': 0.,
            }

        self.ee_ctrl_data = {}
        for joint in BAXTER_GAINS:
            self.ee_ctrl_data[joint] = {
                'prev_err': 0.,
                'cp': 0.,
                'cd': 0.,
                'ci': 0.,
            }

        self.items = items
        self._load_model()

        env = openravepy.Environment()
        self._ikbody = OpenRAVEBody(env, 'baxter', Baxter())

        # Start joints with grippers pointing downward
        self.physics.data.qpos[1:8] = self._calc_ik(START_EE[:3], START_EE[3:7], True)
        self.physics.data.qpos[10:17] = self._calc_ik(START_EE[7:10], START_EE[10:14], False)
        self.physics.forward()

        self.action_inds = {
            ('baxter', 'rArmPose'): np.array(list(range(7))),
            ('baxter', 'rGripper'): np.array([7]),
            ('baxter', 'lArmPose'): np.array(list(range(8, 15))),
            ('baxter', 'lGripper'): np.array([15]),
        }

        if view:
            self._launch_viewer(_CAM_WIDTH, _CAM_HEIGHT)
        else:
            self._viewer = None


    def _load_model(self):
        generate_xml(BASE_VEL_XML, ENV_XML, self.items)
        self.physics = Physics.from_xml_path(ENV_XML)


    def _launch_viewer(self, width, height, title='Main'):
        self._matplot_view_thread = None
        if self.use_glew:
            self._renderer = renderer.NullRenderer()
            self._render_surface = None
            self._viewport = renderer.Viewport(width, height)
            self._window = gui.RenderWindow(width, height, title)
            self._viewer = viewer.Viewer(
                self._viewport, self._window.mouse, self._window.keyboard)
            self._viewer_layout = views.ViewportLayout()
            self._viewer.render()
        else:
            self._viewer = None
            self._matplot_im = None
            self._run_matplot_view()


    def _reload_viewer(self):
        if self._viewer is None or not self.use_glew: return

        if self._render_surface:
            self._render_surface.free()

        if self._renderer:
            self._renderer.release()

        self._render_surface = render.Renderer(
            max_width=_MAX_FRONTBUFFER_SIZE, max_height=_MAX_FRONTBUFFER_SIZE)
        self._renderer = renderer.OffScreenRenderer(
            self.physics.model, self._render_surface)
        self._renderer.components += self._viewer_layout
        self._viewer.initialize(
            self.physics, self._renderer, touchpad=False)
        self._viewer.zoom_to_scene()


    def _render_viewer(self, pixels):
        if self.use_glew:
            with self._window._context.make_current() as ctx:
                ctx.call(
                    self._window._update_gui_on_render_thread, self._window._context.window, pixels)
            self._window._mouse.process_events()
            self._window._keyboard.process_events()
        else:
            if self._matplot_im is not None:
                self._matplot_im.set_data(pixels)
                plt.draw()


    def _run_matplot_view(self):
        self._matplot_view_thread = Thread(target=self._launch_matplot_view)
        self._matplot_view_thread.daemon = True
        self._matplot_view_thread.start()


    def _launch_matplot_view(self):
        self._matplot_im = plt.imshow(self.render(view=False))
        plt.show()


    def get_obs(self):
        obs = []

        # if not len(self.obs_include) or 'image' in self.obs_include:
        #     pixels = self.render(view=False)
        #     obs.extend(pixels.flatten())

        if not len(self.obs_include) or 'joints' in self.obs_include:
            jnts = self.get_joint_angles()
            obs.extend(jnts)

        if not len(self.obs_include) or 'end_effector' in self.obs_include:
            obs.extend(self.get_left_ee_pos())
            obs.extend(self.get_left_ee_rot())
            obs.extend(self.get_right_ee_pos())
            obs.extend(self.get_right_ee_rot())

        for item in self.items:
            if not len(self.obs_include) or item[0] in self.obs_include:
                obs.extend(self.get_item_pos(item[0]))
                obs.extend(self.get_item_rot(item[0]))

        return np.array(obs)


    def get_obs_inds(self, obs_type):
        return self._obs_inds[obs_type]


    def get_arm_section_inds(self, section_name):
        inds = self.get_obs_inds('joints')
        if section_name == 'lArmPose':
            return inds[9:16]
        if section_name == 'lGripper':
            return inds[16:]
        if section_name == 'rArmPose':
            return inds[:7]
        if section_name == 'rGripper':
            return inds[7:8]


    def get_left_ee_pos(self, mujoco_frame=True):
        model = self.physics.model
        ll_gripper_ind = model.name2id('left_gripper_l_finger_tip', 'body')
        lr_gripper_ind = model.name2id('left_gripper_r_finger_tip', 'body')
        pos = (self.physics.data.xpos[ll_gripper_ind] + self.physics.data.xpos[lr_gripper_ind]) / 2
        if not mujoco_frame:
            pos[2] -= MUJOCO_MODEL_Z_OFFSET
        return pos


    def get_right_ee_pos(self, mujoco_frame=True):
        model = self.physics.model
        rr_gripper_ind = model.name2id('right_gripper_r_finger_tip', 'body')
        rl_gripper_ind = model.name2id('right_gripper_l_finger_tip', 'body')
        pos = (self.physics.data.xpos[rr_gripper_ind] + self.physics.data.xpos[rl_gripper_ind]) / 2
        if not mujoco_frame:
            pos[2] -= MUJOCO_MODEL_Z_OFFSET
        return pos


    def get_left_ee_rot(self):
        model = self.physics.model
        l_gripper_ind = model.name2id('left_gripper_base', 'body')
        return self.physics.data.xquat[l_gripper_ind].copy()


    def get_right_ee_rot(self):
        model = self.physics.model
        r_gripper_ind = model.name2id('right_gripper_base', 'body')
        return self.physics.data.xquat[r_gripper_ind].copy()


    def get_item_pos(self, name, mujoco_frame=True):
        model = self.physics.model
        item_ind = model.name2id(name, 'body')
        pos = self.physics.data.xpos[item_ind].copy()
        if not mujoco_frame:
            pos[2] -= MUJOCO_MODEL_Z_OFFSET
        return pos


    def get_item_rot(self, name, convert_to_euler=False):
        model = self.physics.model
        item_ind = model.name2id(name, 'body')
        rot = self.physics.data.xquat[item_ind].copy()
        if convert_to_euler:
            rot = tf.euler_from_quaternion(rot)
        return rot


    def get_cloth_point(self, x, y):
        if not self._cloth_present:
            raise AttributeError('No cloth in model (remember to supply cloth_info).')

        model = self.physics.model
        name = 'B{0}_{1}'.format(x, y)
        if name in self._ind_cache:
            point_ind = self._ind_cache[name]
        else:
            point_ind = model.name2id(name, 'body')
            self._ind_cache[name] = point_ind
        return self.physics.data.xpos[point_ind]


    def get_cloth_points(self):
        if not self._cloth_present:
            raise AttributeError('No cloth in model (remember to supply cloth_info).')

        if not self._cloth_present: return []
        points_inds = []
        model = self.physics.model
        for x in range(self.cloth_length):
            for y in range(self.cloth_width):
                name = 'B{0}_{1}'.format(x, y)
                points_inds.append(model.name2id(name, 'body'))
        return self.physics.data.xpos[points_inds]


    def get_joint_angles(self):
        return self.physics.data.qpos[1:].copy()


    def get_arm_joint_angles(self):
        inds = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16]
        return self.physics.data.qpos[inds]


    def get_gripper_joint_angles(self):
        inds = [8, 17]
        return self.physics.data.qpos[inds]


    def _get_joints(self, act_index):
        if act_index in self._joint_map_cache:
            return self._joint_map_cache[act_index]

        res = []
        for name, attr in self.action_inds:
            inds = self.action_inds[name, attr]
            # Actions have a single gripper command, but MUJOCO uses two gripper joints
            if act_index in inds:
                if attr == 'lGripper':
                    res = [('left_gripper_l_finger_joint', 1), ('left_gripper_r_finger_joint', -1)]
                elif attr == 'rGripper':
                    res = [('right_gripper_r_finger_joint', 1), ('right_gripper_l_finger_joint', -1)]
                elif attr == 'lArmPose':
                    arm_ind = inds.tolist().index(act_index)
                    res = [(MUJOCO_JOINT_ORDER[9+arm_ind], 1)]
                elif attr == 'rArmPose':
                    arm_ind = inds.tolist().index(act_index)
                    res = [(MUJOCO_JOINT_ORDER[arm_ind], 1)]

        self._joint_map_cache[act_index] = res
        return res


    def get_action_meanings(self):
        # For discrete action mode
        return ['NOOP', 'RIGHT_EE_FORWARD', 'RIGHT_EE_BACK', 'RIGHT_EE_LEFT', 'RIGHT_EE_RIGHT',
                'RIGHT_EE_UP', 'RIGHT_EE_DOWN', 'RIGHT_EE_OPEN', 'RIGHT_EE_CLOSE',
                'LEFT_EE_FORWARD', 'LEFT_EE_BACK', 'LEFT_EE_LEFT', 'LEFT_EE_RIGHT',
                'LEFT_EE_UP', 'LEFT_EE_DOWN', 'LEFT_EE_OPEN', 'LEFT_EE_CLOSE']


    def move_right_gripper_forward(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[0] = 0.01
        return self.step(act, mode='end_effector_pos')


    def move_right_gripper_backward(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[0] = -0.01
        return self.step(act, mode='end_effector_pos')


    def move_right_gripper_left(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[1] = 0.01
        return self.step(act, mode='end_effector_pos')


    def move_right_gripper_right(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[1] = -0.01
        return self.step(act, mode='end_effector_pos')


    def move_right_gripper_up(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[2] = 0.01
        return self.step(act, mode='end_effector_pos')


    def move_right_gripper_down(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[2] = -0.01
        return self.step(act, mode='end_effector_pos')

    def open_right_gripper(self):
        act = np.zeros(8)
        act[3] = 0.02
        return self.step(act, mode='end_effector_pos')


    def close_right_gripper(self):
        act = np.zeros(8)
        act[3] = 0
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_forward(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[4] = 0.01
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_backward(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[4] = -0.01
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_left(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[5] = 0.01
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_right(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[5] = -0.01
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_up(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[6] = 0.01
        return self.step(act, mode='end_effector_pos')


    def move_left_gripper_down(self):
        grip_jnts = self.get_gripper_joint_angles()

        act = np.zeros(8)
        act[3] = grip_jnts[0]
        act[7] = grip_jnts[1]
        act[6] = -0.01
        return self.step(act, mode='end_effector_pos')

    def open_left_gripper(self):
        act = np.zeros(8)
        act[7] = 0.02
        return self.step(act, mode='end_effector_pos')


    def close_left_gripper(self):
        act = np.zeros(8)
        act[7] = 0
        return self.step(act, mode='end_effector_pos')


    def _step_joint(self, joint, error):
        ctrl_data = self.ctrl_data[joint]
        gains = BAXTER_GAINS[joint]
        dt = MJC_TIME_DELTA
        de = error - ctrl_data['prev_err']
        ctrl_data['cp'] = error
        ctrl_data['cd'] = de / dt
        ctrl_data['ci'] += error * dt
        ctrl_data['prev_err'] = error
        return gains[0] * ctrl_data['cp'] + \
               gains[1] * ctrl_data['cd'] + \
               gains[2] * ctrl_data['ci']


    def activate_cloth_eq(self):
        for i in range(self.cloth_length):
            for j in range(self.cloth_width):
                pnt = self.get_cloth_point(i, j)
                right_ee = self.get_right_ee_pos()
                left_ee = self.get_left_ee_pos()
                right_grip, left_grip = self.get_gripper_joint_angles()

                r_eq_name = 'right{0}_{1}'.format(i, j)
                l_eq_name = 'left{0}_{1}'.format(i, j)
                if r_eq_name in self._ind_cache:
                    r_eq_ind = self._ind_cache[r_eq_name]
                else:
                    r_eq_ind = self.physics.model.name2id(r_eq_name, 'equality')
                    self._ind_cache[r_eq_name] = r_eq_ind

                if l_eq_name in self._ind_cache:
                    l_eq_ind = self._ind_cache[l_eq_name]
                else:
                    l_eq_ind = self.physics.model.name2id(l_eq_name, 'equality')
                    self._ind_cache[l_eq_name] = l_eq_ind

                if np.all(np.abs(pnt - right_ee) < GRASP_THRESHOLD) and right_grip < 0.015:
                    # if not self.physics.model.eq_active[r_eq_ind]:
                    #     self._shift_cloth_to_grip(right_ee, (i, j))
                    self.physics.model.eq_active[r_eq_ind] = True
                    print('Activated right equality'.format(i, j))
                # else:
                #     self.physics.model.eq_active[r_eq_ind] = False

                if np.all(np.abs(pnt - left_ee) < GRASP_THRESHOLD) and left_grip < 0.015:
                    # if not self.physics.model.eq_active[l_eq_ind]:
                    #     self._shift_cloth_to_grip(left_ee, (i, j))
                    self.physics.model.eq_active[l_eq_ind] = True
                    print('Activated left equality {0} {1}'.format(i, j))
                # else:
                #     self.physics.model.eq_active[l_eq_ind] = False


    def _shift_cloth_to_grip(self, ee_pos, point_xy):
        point_pos = self.get_cloth_point(point_xy[0], point_xy[1])
        cloth_disp = ee_pos - point_pos
        self.physics.data.qpos[19:22] += cloth_disp
        self.physics.forward()


    def _clip_joint_angles(self, r_jnts, r_grip, l_jnts, l_grip):
        DOF_limits = self._ikbody.env_body.GetDOFLimits()
        left_DOF_limits = (DOF_limits[0][2:9]+0.001, DOF_limits[1][2:9]-0.001)
        right_DOF_limits = (DOF_limits[0][10:17]+0.001, DOF_limits[1][10:17]-0.001)

        if r_grip[0] < 0:
            r_grip[0] = 0
        if r_grip[0] > 0.02:
            r_grip[0] = 0.02
        if l_grip[0] < 0:
            l_grip[0] = 0
        if l_grip[0] > 0.02:
            l_grip[0] = 0.02

        for i in range(7):
            if l_jnts[i] < left_DOF_limits[0][i]:
                l_jnts[i] = left_DOF_limits[0][i]
            if l_jnts[i] > left_DOF_limits[1][i]:
                l_jnts[i] = left_DOF_limits[1][i]
            if r_jnts[i] < right_DOF_limits[0][i]:
                r_jnts[i] = right_DOF_limits[0][i]
            if r_jnts[i] > right_DOF_limits[1][i]:
                r_jnts[i] = right_DOF_limits[1][i]


    def _calc_ik(self, pos, quat, use_right=True):
        arm_jnts = self.get_arm_joint_angles()
        grip_jnts = self.get_gripper_joint_angles()
        self._clip_joint_angles(arm_jnts[:7], grip_jnts[:1], arm_jnts[7:], grip_jnts[1:])

        dof_map = {
            'rArmPose': arm_jnts[:7],
            'rGripper': grip_jnts[0],
            'lArmPose': arm_jnts[7:],
            'lGripper': grip_jnts[1],
        }

        manip_name = 'right_arm' if use_right else 'left_arm'
        trans = np.zeros((4, 4))
        trans[:3, :3] = openravepy.matrixFromQuat(quat)[:3,:3]
        trans[:3, 3] = pos
        trans[3, 3] = 1

        jnt_cmd = self._ikbody.get_close_ik_solution(manip_name, trans, dof_map)

        return jnt_cmd


    def step(self, action, mode=None, debug=False):
        if mode is None:
            mode = self.ctrl_mode

        cmd = np.zeros((18))
        abs_cmd = np.zeros((18))

        r_grip = 0
        l_grip = 0

        if mode == 'joint_angle':
            for i in range(len(action)):
                jnts = self._get_joints(i)
                for jnt in jnts:
                    cmd_angle = jnt[1] * action[i]
                    ind = MUJOCO_JOINT_ORDER.index(jnt[0])
                    abs_cmd[ind] = cmd_angle
            r_grip = action[7]
            l_grip = action[15]

        elif mode == 'end_effector':
            # Action Order: ee_right_pos, ee_right_quat, ee_right_grip, ee_left_pos, ee_left_quat, ee_left_grip
            cur_right_ee_pos = self.get_right_ee_pos()
            cur_right_ee_rot = self.get_right_ee_rot()
            cur_left_ee_pos = self.get_left_ee_pos()
            cur_left_ee_rot = self.get_left_ee_rot()

            target_right_ee_pos = cur_right_ee_pos + action[:3]
            target_right_ee_rot = action[3:7] # cur_right_ee_rot + action[3:7]
            target_left_ee_pos = cur_left_ee_pos + action[8:11]
            target_left_ee_rot = action[11:15] # cur_left_ee_rot + action[11:15]

            # target_right_ee_rot /= np.linalg.norm(target_right_ee_rot)
            # target_left_ee_rot /= np.linalg.norm(target_left_ee_rot)

            start_t = time.time()
            right_cmd = self._calc_ik(target_right_ee_pos,
                                      target_right_ee_rot,
                                      use_right=True)

            left_cmd = self._calc_ik(target_left_ee_pos,
                                     target_left_ee_rot,
                                     use_right=False)
            # print 'IK time:', time.time() - start_t

            if right_cmd is None:
                right_cmd = self.get_arm_joint_angles()[:7]
            if left_cmd is None:
                left_cmd = self.get_arm_joint_angles()[7:]

            abs_cmd[:7] = right_cmd
            abs_cmd[9:16] = left_cmd
            r_grip = action[7]
            l_grip = action[15]

        elif mode == 'end_effector_pos':
            # Action Order: ee_right_pos, ee_right_quat, ee_right_grip, ee_left_pos, ee_left_quat, ee_left_grip
            cur_right_ee_pos = self.get_right_ee_pos()
            cur_left_ee_pos = self.get_left_ee_pos()

            target_right_ee_pos = cur_right_ee_pos + action[:3]
            target_right_ee_rot = self.get_right_ee_rot()
            target_left_ee_pos = cur_left_ee_pos + action[4:7]
            target_left_ee_rot = self.get_left_ee_rot()

            start_t = time.time()
            right_cmd = self._calc_ik(target_right_ee_pos,
                                      target_right_ee_rot,
                                      use_right=True)

            left_cmd = self._calc_ik(target_left_ee_pos,
                                     target_left_ee_rot,
                                     use_right=False)
            # print 'IK time:', time.time() - start_t

            if right_cmd is None:
                right_cmd = self.get_arm_joint_angles()[:7]
            if left_cmd is None:
                left_cmd = self.get_arm_joint_angles()[7:]

            abs_cmd[:7] = right_cmd
            abs_cmd[9:16] = left_cmd
            r_grip = action[3]
            l_grip = action[7]

        elif mode == 'discrete_pos':
            if action == 1: return self.move_right_gripper_forward()
            if action == 2: return self.move_right_gripper_backward()
            if aciton == 3: return self.move_right_gripper_left()
            if action == 4: return self.move_right_gripper_right()
            if action == 5: return self.move_right_gripper_up()
            if action == 6: return self.move_right_gripper_down()
            if action == 7: return self.open_right_gripper()
            if action == 8: return self.close_right_gripper()

            if action == 9: return self.move_left_gripper_forward()
            if action == 10: return self.move_left_gripper_backward()
            if aciton == 11: return self.move_left_gripper_left()
            if action == 12: return self.move_left_gripper_right()
            if action == 13: return self.move_left_gripper_up()
            if action == 14: return self.move_left_gripper_down()
            return self.get_obs(), self.compute_reward(), False, {}

        start_t = time.time()
        for t in range(MJC_DELTAS_PER_STEP / 4):
            error = abs_cmd - self.physics.data.qpos[1:19]
            cmd = 7e1 * error
            cmd[7] = 20 if r_grip > 0.0175 else -75
            cmd[8] = -cmd[7]
            cmd[16] = 20 if l_grip > 0.0175 else -75
            cmd[17] = -cmd[16]
            self.physics.set_control(cmd)
            self.physics.step()
        # print 'Step time:', time.time() - start_t

        # if self._cloth_present:
        #     self.activate_cloth_eq()

        if debug:
            print('\n')
            print('Joint Errors:', abs_cmd - self.physics.data.qpos[1:19])
            print('EE Position', self.get_right_ee_pos(), self.get_left_ee_pos())
            print('EE Quaternion', self.get_right_ee_rot(), self.get_left_ee_rot())
            corner1 = self.get_item_pos('B0_0')
            corner2 = self.get_item_pos('B0_{0}'.format(self.cloth_width-1))
            corner3 = self.get_item_pos('B{0}_0'.format(self.cloth_length-1))
            corner4 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width-1))
            print('Cloth corners:', corner1, corner2, corner3, corner4)

        return self.get_obs(), self.compute_reward(), False, {}


    def render(self, height=_CAM_HEIGHT, width=_CAM_WIDTH, camera_id=-1, overlays=(),
             depth=False, scene_option=None, view=True):
        start_t = time.time()
        pixels = self.physics.render(height, width, camera_id, overlays, depth, scene_option)
        # print 'Pixels time:', time.time() - start_t
        if view and self.use_viewer:
            start_t = time.time()
            self._render_viewer(pixels)
            # print 'View time:', time.time() - start_t
        return pixels


    def reset(self):
        self.physics.reset()
        if self._viewer is not None:
            self._reload_viewer()

        self.ctrl_data = {}
        self.cur_time = 0.
        self.prev_time = 0.
        for joint in BAXTER_GAINS:
            self.ctrl_data[joint] = {
                'prev_err': 0.,
                'cp': 0.,
                'cd': 0.,
                'ci': 0.,
            }


    @classmethod
    def init_from_plan(cls, plan, view=True):
        items = []
        for p in plan.params.valuyes():
            if p.is_symbol(): continue
            param_xml = get_param_xml(p)
            if param_xml is not None:
                items.append(param_xml)
        return cls.__init__(view, items)


    def sim_from_plan(self, plan, t):
        model  = self.physics.model
        xpos = model.body_pos.copy()
        xquat = model.body_quat.copy()
        param = list(plan.params.values())

        for param_name in plan.params:
            param = plan.params[param_name]
            if param.is_symbol(): continue
            if param._type != 'Robot':
                param_ind = model.name2id(param.name, 'body')
                if param_ind == -1: continue

                pos = param.pose[:, t]
                xpos[param_ind] = pos + np.array([0, 0, MUJOCO_MODEL_Z_OFFSET])
                if hasattr(param, 'rotation'):
                    rot = param.rotation[:, t]
                    xquat[param_ind] = T.quaternion_from_euler(rot)

        model.body_pos = xpos
        model.body_quat = xquat

        baxter = plan.params['baxter']
        self.physics.data.qpos = np.zeros(19,1)
        self.physics.data.qpos[1:8] = baxter.rArmPose[:, t]
        self.physics.data.qpos[8] = baxter.rGripper[:, t]
        self.physics.data.qpos[9] = -baxter.rGripper[:, t]
        self.physics.data.qpos[10:17] = baxter.lArmPose[:, t]
        self.physics.data.qpos[17] = baxter.lGripper[:, t]
        self.physics.data.qpos[18] = -baxter.lGripper[:, t]

        self.physics.forward()


    def mp_state_from_sim(self, plan):
        X = np.zeros(plan.symbolic_bound)
        for param_name, attr_name in plan.state_inds:
            inds = plan.state_inds[param_name, attr_name]
            if param_name in plan.params:
                param = plan.params[param_name]
                if param_name == 'baxter':
                    pass
                elif not param.is_symbol():
                    if attr_name == 'pose':
                        X[inds] = self.get_item_pos(param_name)
                    elif attr_name == 'rotation':
                        X[inds] = self.get_item_rot(param_name, convert_to_euler=True)




    def jnt_ctrl_from_plan(self, plan, t):
        baxter = plan.params['baxter']
        lArmPose = baxter.lArmPose[:, t]
        lGripper = baxter.lGripper[:, t]
        rArmPose = baxter.rArmPose[:, t]
        rGripper = baxter.rGripper[:, t]
        ctrl = np.r_[rArmPose, rGripper, -rGripper, lArmPose, lGripper, -lGripper]
        return self.step(joint_angles=ctrl)


    def run_plan(self, plan):
        self.reset()
        obs = []
        for t in range(plan.horizon):
            obs.append(self.jnt_ctrl_from_plan(plan, t))

        return obs


    def close(self):
        self.active = False
        if self._viewer is not None and self.use_glew:
            self._viewer.close()
            self._viewer = None
        self.physics.free()


    def seed(self, seed=None):
        pass


    def list_joint_info(self):
        for i in range(self.physics.model.njnt):
            print('\n')
            print('Jnt ', i, ':', self.physics.model.id2name(i, 'joint'))
            print('Axis :', self.physics.model.jnt_axis[i])
            print('Dof adr :', self.physics.model.jnt_dofadr[i])
            body_id = self.physics.model.jnt_bodyid[i]
            print('Body :', self.physics.model.id2name(body_id, 'body'))
            print('Parent body :', self.physics.model.id2name(self.physics.model.body_parentid[body_id], 'body'))


    def compute_reward(self):
        start_t = time.time()
        state = self.check_cloth_state()

        if NO_CLOTH in state: return 0

        if TWO_FOLD in state: return 1e3

        reward = 0

        ee_right_pos = self.get_right_ee_pos()
        ee_left_pos = self.get_left_ee_pos()

        corner1 = self.get_item_pos('B0_0')
        corner2 = self.get_item_pos('B0_{0}'.format(self.cloth_width-1))
        corner3 = self.get_item_pos('B{0}_0'.format(self.cloth_length-1))
        corner4 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width-1))
        corners = [corner1, corner2, corner3, corner4]


        min_right_dist = min([np.linalg.norm(ee_right_pos-corners[i]) for i in range(4)])
        min_left_dist = min([np.linalg.norm(ee_left_pos-corners[i]) for i in range(4)])

        if ONE_FOLD in state:
            reward += 1e2
            if self.cloth_length % 2:
                mid1 = self.get_item_pos('B{0}_0'.format(self.cloth_length // 2))
                mid2 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length // 2, self.cloth_width-1))
            else:
                mid1 = (self.get_item_pos('B{0}_0'.format(self.cloth_length // 2)-1) \
                        + self.get_item_pos('B{0}_0'.format(self.cloth_length // 2))) / 2.0
                mid2 = (self.get_item_pos('B{0}_{1}'.format(self.cloth_length // 2 - 1, self.cloth_width-1)) \
                        + self.get_item_pos('B{0}_{1}'.format(self.cloth_length // 2, self.cloth_width-1))) / 2.0

            if self.cloth_width % 2:
                mid3 = self.get_item_pos('B0_{0}'.format(self.cloth_width // 2))
                mid4 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width // 2))
            else:
                mid3 = (self.get_item_pos('B0_{0}'.format(self.cloth_width // 2)-1) \
                        + self.get_item_pos('B0_{0}'.format(self.cloth_width // 2))) / 2.0
                mid4 = (self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width // 2 - 1)) \
                        + self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width // 2))) / 2.0

            min_dist = min([
                            np.linalg.norm(corner1-ee_left_pos) + np.linalg.norm(mid3-ee_right_pos),
                            np.linalg.norm(corner1-ee_right_pos) + np.linalg.norm(mid3-ee_left_pos),
                            np.linalg.norm(corner3-ee_left_pos) + np.linalg.norm(mid4-ee_right_pos),
                            np.linalg.norm(corner3-ee_right_pos) + np.linalg.norm(mid4-ee_left_pos),
                           ])
            reward -= min_dist
            reward -= 1e1 * np.linalg.norm(corner1 - corner3)
            reward -= 1e1 * np.linalg.norm(mid3 - mid4)
            reward += 1e1 * (0.75 * self.cloth_spacing - np.linalg.norm(corner1 - mid3))
            reward += 1e1 * (0.75 * self.cloth_spacing - np.linalg.norm(corner3 - mid4))

        elif LENGTH_GRASP in state:
            reward += 5e1

            mid1 = self.get_item_pos('B{0}_0'.format(int((self.cloth_length - 1.5) // 2)))
            mid2 = self.get_item_pos('B{0}_0'.format(int((self.cloth_length - 1.5) // 2)))
            mid3 = self.get_item_pos('B{0}_{1}'.format(int((self.cloth_length - 1.5) // 2), self.cloth_width-1))
            mid4 = self.get_item_pos('B{0}_{1}'.format(int((self.cloth_length - 1.5) // 2), self.cloth_width-1))

            mid5 = self.get_item_pos('B0_{0}'.format(int((self.cloth_width - 1.5) // 2)))
            mid6 = self.get_item_pos('B0_{0}'.format(int((self.cloth_width + 1.5) // 2)))
            mid7 = self.get_item_pos('B{0}_{1}'.format(self.cloth_width-1, int((self.cloth_length - 1.5) // 2)))
            mid8 = self.get_item_pos('B{0}_{1}'.format(self.cloth_width-1, int((self.cloth_length + 1.5) // 2)))
            min_dist = min([
                            np.linalg.norm(corner1-ee_left_pos) + np.linalg.norm(corner3-ee_right_pos),
                            np.linalg.norm(corner1-ee_right_pos) + np.linalg.norm(corner3-ee_left_pos),
                            np.linalg.norm(corner2-ee_left_pos) + np.linalg.norm(corner4-ee_right_pos),
                            np.linalg.norm(corner2-ee_right_pos) + np.linalg.norm(corner4-ee_left_pos),
                           ])
            reward -= 1e1 * min_dist
            reward -= 1e1 * np.linalg.norm(corner1[:2] - corner2[:2])
            reward -= 1e1 * np.linalg.norm(corner3[:2] - corner4[:2])
            reward += 1e1 * np.linalg.norm(corner1[:2] - corner3[:2])
            reward += 1e1 * np.linalg.norm(corner2[:2] - corner4[:2])
            reward -= 5e0 * np.linalg.norm(mid5[:2] - mid6[:2])
            reward -= 5e0 * np.linalg.norm(mid7[:2] - mid8[:2])

        else:
            right_most_corner = min(corners, key=lambda c: c[0])
            left_most_corner = max(corners, key=lambda c: c[0])

            r_corner_id = np.argmin([np.linalg.norm(c - right_most_corner) for c in corners])
            l_corner_id = np.argmin([np.linalg.norm(c - left_most_corner) for c in corners])
            reward -= 1e1 * np.linalg.norm(ee_left_pos - left_most_corner)
            if right_most_corner[0] > 0.1:
                reward -= 1e1 * np.linalg.norm(ee_left_pos - right_most_corner)
                reward -= 1e1 * right_most_corner[0]
            elif left_most_corner[0] > -0.1:
                reward += 5
                next_corner1 = corners[(-2 + l_corner_id) % 4]
                next_corner2 = corners[(-2 + r_corner_id) % 4]
                if next_corner1[0] < 0.1:
                    reward -= 1e1 * np.linalg.norm(ee_left_pos - left_most_corner)
                    reward -= 1e1 * np.linalg.norm(ee_right_pos - next_corner1)
                elif next_corner2[0] > -0.1:
                    reward -= 1e1 * np.linalg.norm(ee_right_pos - right_most_corner)
                    reward -= 1e1 * np.linalg.norm(ee_left_pos - next_corner2)
                else:
                    reward -= 1e1 * np.linalg.norm(ee_left_pos - next_corner1)
                    reward -= 1e1 * next_corner1[0]

            else:
                reward -= 4e1 * np.linalg.norm(ee_right_pos - left_most_corner)
                reward += 1e1 * left_most_corner[0]

        # print 'Reward calculation time:', time.time() - start_t
        return reward


    def check_cloth_state(self):
        if not self._cloth_present: return [NO_CLOTH]

        state = []

        # Check full fold
        corner1 = self.get_item_pos('B0_0')
        corner2 = self.get_item_pos('B0_{0}'.format(self.cloth_width-1))
        corner3 = self.get_item_pos('B{0}_0'.format(self.cloth_length-1))
        corner4 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width-1))

        corners = [corner1, corner2, corner3, corner4]
        check1 = all([all([np.max(np.abs(corners[i][:2] - corners[j][:2])) < 0.04 for j in range(i+1, 4)]) for i in range(4)])

        mid1 = self.get_item_pos('B{0}_0'.format(int((self.cloth_length - 1.5) // 2)))
        mid2 = self.get_item_pos('B{0}_0'.format(int((self.cloth_length + 1.5) // 2)))
        mid3 = self.get_item_pos('B{0}_{1}'.format(int((self.cloth_length - 1.5) // 2), self.cloth_width-1))
        mid4 = self.get_item_pos('B{0}_{1}'.format(int((self.cloth_length + 1.5) // 2), self.cloth_width-1))
        mids = [mid1, mid2, mid3, mid4]
        check2 = all([all([np.max(np.abs(mids[i][:2] - mids[j][:2])) < 0.04 for j in range(i+1, 4)]) for i in range(4)])

        mid5 = self.get_item_pos('B0_{0}'.format(int((self.cloth_width - 1.5) // 2)))
        mid6 = self.get_item_pos('B0_{0}'.format(int((self.cloth_width + 1.5) // 2)))
        mid7 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, int((self.cloth_width - 1.5) // 2)))
        mid8 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, int((self.cloth_width + 1.5) // 2)))
        mids = [mid5, mid6, mid7, mid8]
        check2 = all([all([np.max(np.abs(mids[i][:2] - mids[j][:2])) < 0.04 for j in range(i+1, 4)]) for i in range(4)])

        check3 = np.linalg.norm(corner1[:2] - mid1[:2]) > 0.75 * self.cloth_spacing and \
                 np.linalg.norm(corner1[:2] - mid5[:2]) > 0.75 * self.cloth_spacing and \
                 np.linalg.norm(mid1[:2] - mid5[:2]) > 0.75 * self.cloth_spacing

        if check1 and check2 and check3: state.append(TWO_FOLD)

        # Check length-wise fold
        corner1 = self.get_item_pos('B0_0')
        corner2 = self.get_item_pos('B0_{0}'.format(self.cloth_width-1))
        check1 = np.max(np.abs(corner1[:2] - corner2[:2])) < 0.04

        mid1 = self.get_item_pos('B0_{0}'.format(int((self.cloth_width - 1.5) // 2)))
        mid2 = self.get_item_pos('B0_{0}'.format(int((self.cloth_width + 1.5) // 2)))
        check2 = np.max(np.abs(mid1[:2] - mid2[:2])) < 0.02

        corner3 = self.get_item_pos('B{0}_0'.format(self.cloth_length-1))
        corner4 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width-1))
        check3 = np.max(np.abs(corner3[:2] - corner4[:2])) < 0.04

        mid3 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, int((self.cloth_width - 1.5) // 2)))
        mid4 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, int((self.cloth_width + 1.5) // 2)))
        check4 = np.max(np.abs(mid3[:2] - mid4[:2])) < 0.04

        check5 = np.linalg.norm(corner1[:2] - mid1[:2]) > 0.75 * self.cloth_spacing and \
                 np.linalg.norm(corner3[:2] - mid3[:2]) > 0.75 * self.cloth_spacing

        if check1 and check2 and check3 and check4 and check5: state.append(ONE_FOLD)

        # Check twist-fold
        corner1 = self.get_item_pos('B0_0')
        corner2 = self.get_item_pos('B0_{0}'.format(self.cloth_width-1))
        corner3 = self.get_item_pos('B{0}_0'.format(self.cloth_length-1))
        corner4 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width-1))

        dist1 = np.linalg.norm(corner1 - corner4)
        dist2 = np.linalg.norm(corner2 - corner3)
        if dist1 > dist2:
            check1 = dist1 > 0.9 * (self.cloth_spacing*np.sqrt(self.cloth_width**2+self.cloth_length**2))
            check2 = np.abs(corner1[0] - corner4[0]) < 0.08

            far_x_pos = 0.8 * (self.cloth_length * self.cloth_width / dist1)
            check3 = corner3[0] - corner1[0] > far_x_pos and corner2[0] - corner4[0] > far_x_pos
        else:
            check1 = dist2 > 0.9 * (self.cloth_spacing*np.sqrt(self.cloth_width**2+self.cloth_length**2))
            check2 = np.abs(corner2[0] - corner3[0]) < 0.08

            far_x_pos = 0.8 * (self.cloth_length * self.cloth_width / dist1)
            check3 = corner1[0] - corner3[0] > far_x_pos and corner4[0] - corner2[0] > far_x_pos
        if check1 and check2 and check3: state.append('TWIST_FOLD')


        # Check two corner grasp
        corner1 = self.get_item_pos('B0_0')
        corner2 = self.get_item_pos('B0_{0}'.format(self.cloth_width-1))
        corner3 = self.get_item_pos('B{0}_0'.format(self.cloth_length-1))
        corner4 = self.get_item_pos('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width-1))

        corners = [corner1, corner2, corner3, corner4]

        ee_left_pos = self.get_left_ee_pos()
        ee_right_pos = self.get_right_ee_pos()
        grips = self.get_gripper_joint_angles()

        check1 = np.linalg.norm(ee_right_pos - corner1) < 0.02 and grips[0] < 0.04 and np.linalg.norm(ee_left_pos - corner3) < 0.02 and grips[1] < 0.04
        check2 = np.linalg.norm(ee_left_pos - corner1) < 0.02 and grips[1] < 0.04 and np.linalg.norm(ee_right_pos - corner3) < 0.02 and grips[0] < 0.04

        check3 = np.linalg.norm(ee_right_pos - corner2) < 0.02 and grips[0] < 0.04 and np.linalg.norm(ee_left_pos - corner4) < 0.02 and grips[1] < 0.04
        check4 = np.linalg.norm(ee_left_pos - corner2) < 0.02 and grips[1] < 0.04 and np.linalg.norm(ee_right_pos - corner4) < 0.02 and grips[0] < 0.04

        if check1 or check2 or check3 or check4: state.append(LENGTH_GRASP)

        return state
