import xml.etree.ElementTree as ET
import robosuite
from robosuite.controllers import load_controller_config
cur_objs = ['cereal', 'milk', 'can', 'bread']
ctrl_mode = "JOINT_POSITION"
true_mode = 'JOINT'

controller_config = load_controller_config(default_controller=ctrl_mode)
if ctrl_mode.find('JOINT') >= 0:
    controller_config['kp'] = [7500, 6500, 6500, 6500, 6500, 6500, 12000]
    controller_config['output_max'] = 0.2
    controller_config['output_min'] = -0.2
else:
    controller_config['kp'] = 5000 # [8000, 8000, 8000, 4000, 4000, 4000]
    controller_config['input_max'] = 0.2 #[0.05, 0.05, 0.05, 4, 4, 4]
    controller_config['input_min'] = -0.2 # [-0.05, -0.05, -0.05, -4, -4, -4]
    controller_config['output_max'] = 0.02 # [0.1, 0.1, 0.1, 2, 2, 2]
    controller_config['output_min'] = -0.02 # [-0.1, -0.1, -0.1, -2, -2, -2]

visual = False # len(os.environ.get('DISPLAY', '')) > 0
has_render = visual
obj_mode = 0 if len(cur_objs) > 1 else 2
env = robosuite.make(
    "PickPlace",
    robots=["Sawyer"],             # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    controller_configs=controller_config,   # each arm is controlled using OSC
    #has_renderer=True,                      # on-screen rendering
    has_renderer=has_render,                      # on-screen rendering
    render_camera="frontview",              # visualize the "frontview" camera
    has_offscreen_renderer=(not has_render),           # no off-screen rendering
    control_freq=50,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=True,                   # no observations needed
    use_camera_obs=False,                   # no observations needed
    single_object_mode=obj_mode,
    object_type=cur_objs[0],
    ignore_done=True,
    reward_shaping=True,
    initialization_noise={'magnitude': 0., 'type': 'gaussian'},
    render_gpu_device_id=0,
)
import ipdb; ipdb.set_trace()

