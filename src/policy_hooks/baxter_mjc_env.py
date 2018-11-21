import xml.etree.ElementTree as xml

BASE_POS_XML = '../models/baxter/mujoco/baxter_mujoco_pos.xml'
BASE_MOTOR_XML = '../models/baxter/mujoco/baxter_mujoco.xml'
ENV_XML = 'policy_hooks/mujoco/current_baxter_env.xml'

MUJOCO_JOINT_ORDER = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_e2', 'left_w0', 'left_w1', 'left_w2', 'left_gripper_l_finger_joint', 'left_gripper_r_finger_joint'\
                      'right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2', 'right_gripper_l_finger_joint', 'right_gripper_r_finger_joint']

BAXTER_GAINS = {
    'left_s0': (700., 0.01, 25.),
    'left_s1': (10000., 100., 100.),
    'left_e0': (4500., 35., 1.),
    'left_e1': (5500, 60, 2),
    'left_w0': (1000, 30, 0.01),
    'left_w1': (900, 0.1, 0.01),
    'left_w2': (1000, 0.1, 0.01),
    'left_gripper_l_finger_joint': (1000, 0.1, 0.01),
    'left_gripper_r_finger_joint': (1000, 0.1, 0.01),

    'right_s0': (700., 0.01, 100.),
    'right_s1': (10000., 100., 100.),
    'right_e0': (4500., 35., 1.),
    'right_e1': (5500, 60, 2),
    'right_w0': (1000, 30, 0.01),
    'right_w1': (900, 0.1, 0.01),
    'right_w2': (1000, 0.1, 0.01),
    'right_gripper_l_finger_joint': (1000, 0.1, 0.01),
    'right_gripper_r_finger_joint': (1000, 0.1, 0.01),
}

MJC_TIME_DELTA = 0.1
MJC_DELTAS_PER_STEP = int(1. // MJC_TIME_DELTA)
MUJOCO_MODEL_Z_OFFSET = -0.706

N_CONTACT_LIMIT = 12

class BaxterMJCEnv(object):
    def __init__(self, view=False, items=[]):
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
        self.items = items
        self._load_model()

        if view:
            self._launch_viewer()


    def _generate_xml(self):
        base_xml = xml.parse(BASE_MOTOR_XML)
        root = base_xml.getroot()
        worldbody = root.find('worldbody')
        active_ts = (0, plan.horizon)
        contacts = root.find('contact')
        items = self.items

        for item_body, tag_dict in self.items:
            base_xml.append(item_body)
            if 'contacts' in tag_dict:
                for contact in tag_dict['contacts']:
                    contacts.append(contact)

        base_xml.write(ENV_XML)


    def _load_model(self):
        self._generate_xml()
        self.physics = Physics.from_xml_path(ENV_XML)


    def _launch_viewer(self):
         pass


    def get_left_ee_pose(self):
        model = self.physics.model
        l_gripper_ind = model.mj_name2id('left_gripper_l_finger_tip', 'body')
        return self.physics.data.xpos[l_gripper_ind]


    def get_right_ee_pose(self):
        model = self.physics.model
        r_gripper_ind = model.mj_name2id('right_gripper_r_finger_tip', 'body')
        return self.physics.data.xpos[r_gripper_ind]


    def get_item_pose(self, name):
        model = self.physics.model
        item_ind = model.mj_name2id(name, 'body')
        return self.physics.data.xpos[item_ind]


    def get_joint_angles(self):
        return self.physics.model.qpos[1:].copy()


    def _step_joint(self, joint, error):
        ctrl_data = self.ctrl_data[joint]
        gains = BAXTER_GAINS[joint]
        dt = MJC_TIME_DELTA
        de = error - ctrl_data[joint]['prev_error']
        ctrl_data['cp'] = error
        ctrl_data['cd'] = de / dt
        ctrl_data['ci'] += error * dt
        ctrl_data['prev_err'] = error
        return gains[0] * ctrl_data['cp'] + \
               gains[1] * ctrl_data['cd'] + \
               gains[2] * ctrl_data['ci']


    def step(self, torques=None, joint_angles=None):
        if torques is None and joint_angles is None:
            print 'Baxter MJC Env received null step command.'
            return

        for t in range(MJC_DELTAS_PER_STEP):
            action = np.zeros(len(MUJOCO_JOINT_ORDER))
            if torques is not None:
                for joint in torques:
                    ind = MUJOCO_JOINT_ORDER.index(joint)
                    action[ind] = torques[joint]
            elif joint_angles is not None:
                for joint in MUJOCO_JOINT_ORDER:
                    ind = MUJOCO_JOINT_ORDER.index(joint)
                    current_angle = self.physics.data.qpos[ind+1]
                    cmd_angle = joint_angles[joint]
                    error = cmd_angle - current_angle
                    cmd_torque = self._step_joint(joint, error)
                    aaction[ind] = cmd_torque

            self.physics.set_control(action)
            self.physics.step()

        return self.get_obs()


    def render(self, height=240, width=320, camera_id=-1, overlays=(),
             depth=False, scene_option=None):
        return self.physics.render(height, width, camera_id, overlays, depth, scene_option)


    def reset(self):
        self.physics.reset()
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


    def _set_simulator_from_plan(self, plan, t):
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
                xpos[param_ind] = pos + np.array([0, 0, MUJOCO_MODEL_Z_OFFSET]) + np.array([0, 0, 0.025])
                if hasattr(param, 'rotation'):
                    rot = param.rotation[:, t]
                    xquat[param_ind] = openravepy.quatFromRotationMatrix(OpenRAVEBody.transform_from_obj_pose(pos, rot)[:3,:3])

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


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


    def seed(self, seed=None):
        pass
