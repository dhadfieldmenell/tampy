import numpy as np

from baxter_gym.envs import BaxterMJCEnv


dim = 0.03
mass = 2
config = {
    #'tampy_path': "/home/michaelmcdonald/dependencies/tampy",
    'obs_include': ['forward_image'],
    'include_files': ['robot_info/meshes/local/duplo1.stl'],
    'include_items': [
        {'name': 'block1', 'type': 'box', 'is_fixed': False, 'pos': (0.45, 0.6, -0.02), 'dimensions': (dim, dim, dim), 'mass': mass},
        {'name': 'block2', 'type': 'box', 'is_fixed': False, 'pos': (0.75, 0.35, -0.02), 'dimensions': (dim, dim, dim), 'mass': mass},
        {'name': 'block3', 'type': 'box', 'is_fixed': False, 'pos': (0.75, 0.7, -0.02), 'dimensions': (dim, dim, dim), 'mass': mass},
    ],
    'sim_freq': 75,
    'view': True,
}

env = BaxterMJCEnv.load_config(config)
env.render(camera_id=1, view=True)
env.render(camera_id=1, view=True)

print(env.get_item_pos('block1'))
env.move_left_to_grasp([0.75, 0.7, -0.01], view=True)
env.move_left_to_place([0.75, 0.35, 0.05], view=True)
env.move_left_to_grasp([0.45, 0.6, -0.01], view=True)
env.move_left_to_place([0.75, 0.35, 0.11], view=True)
import ipdb; ipdb.set_trace()

