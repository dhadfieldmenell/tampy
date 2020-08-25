from baxter_gym.envs import HSRRosEnv

config = {
    'tampy_path': "/home/michaelmcdonald/dependencies/tampy",
    'obs_include': ['overhead_camera'],
    'include_files': ['robot_info/meshes/local/duplo1.stl'],
    'include_items': [
        {'name': 'table', 'type': 'box', 'is_fixed': True, 'pos': (1.5, 0, 0.2), 'dimensions': (0.25, 0.5, 0.2)},
        {'name': 'duplo', 'type': 'mesh', 'is_fixed': False, 'pos': (1.25, 0, 0.43), 'mesh_name': 'duplo1'},
    ],
    'view': True,
}

env = HSRRosEnv.load_config(config)
env.render(camera_id=1, view=True)
env.render(camera_id=1, view=True)
