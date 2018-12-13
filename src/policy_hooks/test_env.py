import numpy as np
import time

from policy_hooks.baxter.baxter_mjc_env import BaxterMJCEnv
from policy_hooks.utils.mjc_xml_utils import *


def print_diff(target, qpos):
    print np.r_[target[:8] - qpos[1:9], target[8:] - qpos[10:18]]

def test_move():
    cloth = get_deformable_cloth(4, 3, (1., 0., 0.5))
    env = BaxterMJCEnv(items=[cloth], view=True)
    env.render(camera_id=0)
    import ipdb; ipdb.set_trace()

    # act_one = np.zeros((16,))
    # env.step(act_one)
    # env.render()
    # time.sleep(0.5)
    # print_diff(act_one, env.physics.data.qpos)

    # act_two = np.zeros((16,))
    # act_two[0] = -0.75
    # act_two[9] = 0.75
    # env.step(act_two)
    # env.render(camera_id=1)
    # time.sleep(0.5)
    # print_diff(act_two, env.physics.data.qpos)

    # act_three = np.zeros((16,))
    # act_three[0] = -1.5
    # act_three[9] = 1.5
    # env.step(act_three)
    # env.render(camera_id=1)
    # time.sleep(0.5)
    # print_diff(act_three, env.physics.data.qpos)

    # env.step(act_two)
    # env.render(camera_id=1)
    # time.sleep(0.5)
    # print_diff(act_two, env.physics.data.qpos)

    # env.step(act_one)
    # env.render(camera_id=1)
    # time.sleep(0.5)
    # print_diff(act_one, env.physics.data.qpos[1:19])

    end_pose = np.array([0.7, -1.01026434, -0.078992, 0.90689717, 0.04219482, 1.67547625, -0.1828384, 0., -0.7, -1.06296608, 0.22662184, 1.00532877, -0.10973573, 1.63895279, 0.24181173, 0.])
    env.step(np.array(end_pose / 4.))
    env.render(camera_id=1)
    print_diff(end_pose / 4., env.physics.data.qpos)

    env.step(np.array(end_pose / 2.))
    env.render(camera_id=1)
    print_diff(end_pose / 2., env.physics.data.qpos)

    env.step(3*np.array(end_pose / 4.))
    env.render(camera_id=1)
    print_diff(3*end_pose / 4., env.physics.data.qpos)

    env.step(end_pose)
    env.render(camera_id=1)
    print_diff(end_pose, env.physics.data.qpos)

    env.step(end_pose)
    env.render(camera_id=1)
    print_diff(end_pose, env.physics.data.qpos)

    env.step(end_pose)
    env.render(camera_id=1)
    print_diff(end_pose, env.physics.data.qpos)

    env.close()

test_move()
