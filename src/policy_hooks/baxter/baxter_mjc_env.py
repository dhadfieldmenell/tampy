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


CTRL_MODES = ['joint_angle', 'end_effector']

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
        if self.ctrl_mode == 'end_effector':
            # self.ee_ctrl = BaxterIKController(lambda: self.get_arm_joint_angles())
            env = openravepy.Environment()
            self._ikbody = OpenRAVEBody(env, 'baxter', Baxter())

        self.action_inds = {
            ('baxter', 'rArmPose'): np.array(range(7)),
            ('baxter', 'rGripper'): np.array([7]),
            ('baxter', 'lArmPose'): np.array(range(8, 15)),
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
            obs.extend(self.get_left_ee_pose())
            obs.extend(self.get_left_ee_rot())
            obs.extend(self.get_right_ee_pose())
            obs.extend(self.get_right_ee_rot())

        for item in self.items:
            if not len(self.obs_include) or item[0] in self.obs_include:
                obs.extend(self.get_item_pose(item[0]))
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


    def get_left_ee_pose(self):
        model = self.physics.model
        ll_gripper_ind = model.name2id('left_gripper_l_finger_tip', 'body')
        lr_gripper_ind = model.name2id('left_gripper_r_finger_tip', 'body')
        return (self.physics.data.xpos[ll_gripper_ind] + self.physics.data.xpos[lr_gripper_ind]) / 2


    def get_right_ee_pose(self):
        model = self.physics.model
        rr_gripper_ind = model.name2id('right_gripper_r_finger_tip', 'body')
        rl_gripper_ind = model.name2id('right_gripper_l_finger_tip', 'body')
        return (self.physics.data.xpos[rr_gripper_ind] + self.physics.data.xpos[rl_gripper_ind]) / 2


    def get_left_ee_rot(self):
        model = self.physics.model
        l_gripper_ind = model.name2id('left_gripper_base', 'body')
        return self.physics.data.xquat[l_gripper_ind].copy()


    def get_right_ee_rot(self):
        model = self.physics.model
        r_gripper_ind = model.name2id('right_gripper_base', 'body')
        return self.physics.data.xquat[r_gripper_ind].copy()


    def get_item_pose(self, name, mujoco_frame=True):
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
                right_ee = self.get_right_ee_pose()
                left_ee = self.get_left_ee_pose()
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
                    print 'Activated right equality'.format(i, j)
                # else:
                #     self.physics.model.eq_active[r_eq_ind] = False

                if np.all(np.abs(pnt - left_ee) < GRASP_THRESHOLD) and left_grip < 0.015:
                    # if not self.physics.model.eq_active[l_eq_ind]:
                    #     self._shift_cloth_to_grip(left_ee, (i, j))
                    self.physics.model.eq_active[l_eq_ind] = True
                    print 'Activated left equality {0} {1}'.format(i, j)
                # else:
                #     self.physics.model.eq_active[l_eq_ind] = False


    def _shift_cloth_to_grip(self, ee_pos, point_xy):
        point_pos = self.get_cloth_point(point_xy[0], point_xy[1])
        cloth_disp = ee_pos - point_pos
        xpos = self.physics.data.xpos
        for i in range(self.cloth_length):
            for j in range(self.cloth_width):
                if 'B{0}_{1}'.format(i, j) in self._ind_cache:
                    point_ind = self._ind_cache['B{0}_{1}'.format(i, j)]
                else:
                    point_ind = self.physics.model.name2id('B{0}_{1}'.format(i, j), 'body')
                    self._ind_cache['B{0}_{1}'.format(i, j)] = point_ind
                xpos[point_ind] += cloth_disp
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


    def step(self, action, debug=False):
        cmd = np.zeros((18))
        abs_cmd = np.zeros((18))

        if self.ctrl_mode == 'joint_angle':
            for i in range(len(action)):
                jnts = self._get_joints(i)
                for jnt in jnts:
                    cmd_angle = jnt[1] * action[i]
                    ind = MUJOCO_JOINT_ORDER.index(jnt[0])
                    abs_cmd[ind] = cmd_angle

        elif self.ctrl_mode == 'end_effector':
            # Order: ee_right_pos, ee_right_quat, ee_right_grip, ee_left_pos, ee_left_quat, ee_left_grip
            abs_cmd[:7] = self._calc_ik(action[:3], action[3:7], use_right=True)
            abs_cmd[9:16] = self._calc_ik(action[8:11], action[11:15], use_right=False)

        for t in range(MJC_DELTAS_PER_STEP):
            error = abs_cmd - self.physics.data.qpos[1:19]
            cmd = 6e1 * error
            cmd[7] = 10 if action[7] > 0.0175 else -100
            cmd[8] = -cmd[7]
            cmd[16] = 10 if action[15] > 0.0175 else -100
            cmd[17] = -cmd[16]
            self.physics.set_control(cmd)
            self.physics.step()

        # if self._cloth_present:
        #     self.activate_cloth_eq()

        if debug:
            print '\n'
            print 'Joint Errors:', abs_cmd - self.physics.data.qpos[1:19]
            print 'EE Pose', self.get_right_ee_pose(), self.get_left_ee_pose()
            corner1 = self.get_item_pose('B0_0')
            corner2 = self.get_item_pose('B0_{0}'.format(self.cloth_width-1))
            corner3 = self.get_item_pose('B{0}_0'.format(self.cloth_length-1))
            corner4 = self.get_item_pose('B{0}_{1}'.format(self.cloth_length-1, self.cloth_width-1))
            print 'Cloth corners:', corner1, corner2, corner3, corner4

        return self.get_obs(), 0, False, {}


    def render(self, height=_CAM_HEIGHT, width=_CAM_WIDTH, camera_id=-1, overlays=(),
             depth=False, scene_option=None, view=True):
        pixels = self.physics.render(height, width, camera_id, overlays, depth, scene_option)
        if view and self.use_viewer:
            self._render_viewer(pixels)
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
        param = plan.params.values()

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
                        X[inds] = self.get_item_pose(param_name)
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
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


    def seed(self, seed=None):
        pass
