import numpy as np

import core.util_classes.baxter_constants as baxter_constants
import core.util_classes.baxter_sampling as baxter_sampling

DOWN_ROT = [0, np.pi/2, 0]


def obj_pose_suggester(plan, anum, resample_size=20):
    robot_pose = []

    if anum + 1 < len(plan.actions):
        act, next_act = plan.actions[anum], plan.actions[anum+1]
    else:
        act, next_act = plan.actions[anum], None

    robot = plan.params['baxter']
    robot_body = robot.openrave_body
    start_ts, end_ts = act.active_timesteps
    old_l_arm_pose = robot.lArmPose[:, start_ts].reshape((7, 1))
    old_r_arm_pose = robot.rArmPose[:, start_ts].reshape((7, 1))
    old_pose = robot.pose[:, start_ts].reshape((1, 1))
    if act.name.find("grasp") >= 0 or act.name.find("hold") >= 0:
        gripper_val = np.array([[baxter_constants.GRIPPER_CLOSE_VALUE]])
    else:
        gripper_val = np.array([[baxter_constants.GRIPPER_OPEN_VALUE]])

    robot_body.set_dof({'lArmPose': [0.785, -0.785, 0, 0, 0, 0, 0], 'rArmPose':[-0.785, -0.785, 0, 0, 0, 0, 0], 'lGripper': [0.02], 'rGripper': [0.02]})

    for i in range(resample_size):
        if act.name == "rotate_holding_basket_with_cloth" or act.name == "rotate_holding_basket":
            target_rot = act.params[4]
            init_pos = act.params[2]
            robot_pose.append({'lArmPose': init_pos.lArmPose.copy(), 'rArmPose': init_pos.rArmPose.copy(), 'lGripper': init_pos.lGripper.copy(), 'rGripper': init_pos.rGripper.copy(), 'value': target_rot.value.copy()})

        elif act.name == "rotate_holding_cloth":
            target_rot = act.params[4]
            init_pos = act.params[2]
            robot_pose.append({'lArmPose': init_pos.lArmPose.copy(), 'rArmPose': init_pos.rArmPose.copy(), 'lGripper': init_pos.lGripper.copy(), 'rGripper': init_pos.rGripper.copy(), 'value': target_rot.value.copy()})

        elif next_act != None and (next_act.name == 'basket_grasp' or next_act.name == 'basket_grasp_with_cloth'):
            target = next_act.params[2]
            target_rot = target.rotation[0, 0]
            handle_dist = baxter_constants.BASKET_OFFSET
            offset = np.array([handle_dist*np.cos(target_rot), handle_dist*np.sin(target_rot), 0])
            target_pos = target.value[:, 0]

            next_act.params[1].openrave_body.set_pose(target_pos, target.rotation[:, 0])

            const_dir = [0, 0, .125]
            ee_left = target_pos + offset + const_dir + np.multiply(np.random.sample(3)-[0.5, 0.5, 0], [0.1, 0.1, 0.1])
            ee_right = target_pos - offset + const_dir + np.multiply(np.random.sample(3)-[0.5, 0.5, 0], [0.1, 0.1, 0.1])

            l_arm_pose = robot_body.get_ik_from_pose(ee_left, [target_rot-np.pi/2, np.pi/2, 0], "left_arm")
            r_arm_pose = robot_body.get_ik_from_pose(ee_right, [target_rot-np.pi/2, np.pi/2, 0], "right_arm")
            if not len(l_arm_pose) or not len(r_arm_pose):
                continue
            l_arm_pose = baxter_sampling.closest_arm_pose(l_arm_pose, old_l_arm_pose.flatten()).reshape((7,1))
            r_arm_pose = baxter_sampling.closest_arm_pose(r_arm_pose, old_r_arm_pose.flatten()).reshape((7,1))

            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': l_arm_pose, 'rArmPose': r_arm_pose, 'lGripper': gripper_val, 'rGripper': gripper_val, 'value': old_pose})

        elif next_act != None and (next_act.name == 'basket_putdown' or next_act.name == 'basket_putdown_with_cloth'):
            target = next_act.params[2]
            target_rot = target.rotation[0, 0]
            handle_dist = baxter_constants.BASKET_OFFSET
            offset = np.array([handle_dist*np.cos(target_rot), handle_dist*np.sin(target_rot), 0])
            target_pos = target.value[:, 0]

            next_act.params[1].openrave_body.set_pose(target_pos, target.rotation[:, 0])

            const_dir = [0, 0, .125]
            ee_left = target_pos + offset + const_dir
            ee_right = target_pos - offset + const_dir

            l_arm_pose = robot_body.get_ik_from_pose(ee_left, [target_rot-np.pi/2, np.pi/2, 0], "left_arm")
            r_arm_pose = robot_body.get_ik_from_pose(ee_right, [target_rot-np.pi/2, np.pi/2, 0], "right_arm")
            if not len(l_arm_pose) or not len(r_arm_pose):
                continue
            l_arm_pose = baxter_sampling.closest_arm_pose(l_arm_pose, old_l_arm_pose.flatten()).reshape((7,1))
            r_arm_pose = baxter_sampling.closest_arm_pose(r_arm_pose, old_r_arm_pose.flatten()).reshape((7,1))

            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': l_arm_pose, 'rArmPose': r_arm_pose, 'lGripper': gripper_val, 'rGripper': gripper_val, 'value': old_pose})

        elif next_act != None and next_act.name == 'put_into_washer':
            target = next_act.params[2]
            target_body = next_act.params[1].openrave_body
            target_body.set_pose(target.value[:, 0], target.rotation[:, 0])
            target_pos = target.value[:,0]
            target_rot = target.rotation[:,0]

            random_dir = np.multiply(np.random.sample(3) - [2.0, 2.5, 0.5], [0.2, 0.2, 0.01])
            ee_pos = target_pos + random_dir
            ee_rot = np.array([target_rot[0] - np.pi/2, 0, 0])

            ik_arm_poses = robot_body.get_ik_from_pose(ee_pos, ee_rot, "left_arm")
            if not len(ik_arm_poses):
                continue
            arm_pose = baxter_sampling.closest_arm_pose(ik_arm_poses, old_l_arm_pose.flatten()).reshape((7,1))

            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': arm_pose, 'rArmPose': old_r_arm_pose, 'lGripper': gripper_val, 'rGripper': gripper_val, 'value': old_pose})

        elif next_act != None and next_act.name == 'take_out_washer':
            target = next_act.params[2]
            target_body = next_act.params[1].openrave_body
            target_body.set_pose(target.value[:, 0], target.rotation[:, 0])
            target_pos = target.value[:,0]
            target_rot = target.rotation[:,0]

            random_dir = np.multiply(np.random.sample(3) - [2.0, 2.5, 0.5], [0.2, 0.2, 0.01])
            ee_pos = target_pos + random_dir
            ee_rot = np.array([target_rot[0] - np.pi/2, 0, 0])

            ik_arm_poses = robot_body.get_ik_from_pose(ee_pos, ee_rot, "left_arm")
            if not len(ik_arm_poses):
                continue
            arm_pose = baxter_sampling.closest_arm_pose(ik_arm_poses, old_l_arm_pose.flatten()).reshape((7,1))

            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': arm_pose, 'rArmPose': old_r_arm_pose, 'lGripper': gripper_val, 'rGripper': gripper_val, 'value': old_pose})

        elif next_act != None and next_act.name == 'push_door':
            target = next_act.params[4]
            target_body = next_act.params[1].openrave_body
            target_body.set_pose(target.value[:, 0], target.rotation[:, 0])
            target_body.set_dof({'door': target.door[:, 0]})
            target_pos = target_body.env_body.GetLink('washer_door').GetTransform()[:3,3]

            random_dir = np.multiply(np.random.sample(3) - [0.5, 1, -4.0], [0.01, 0.05, 0.1])
            ee_pos = target_pos + random_dir
            ee_rot = np.array([np.pi/2, 0, 0])
            ik_arm_poses = robot_body.get_ik_from_pose(ee_pos, ee_rot, "left_arm")
            if not len(ik_arm_poses):
                continue
            arm_pose = baxter_sampling.closest_arm_pose(ik_arm_poses, old_l_arm_pose.flatten()).reshape((7,1))

            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': arm_pose, 'rArmPose': old_r_arm_pose, 'lGripper': gripper_val, 'rGripper': gripper_val, 'value': old_pose})

        elif act.name == 'take_out_of_washer':
            target = act.params[2]
            target_body = act.params[1].openrave_body
            target_body.set_pose(target.value[:, 0], target.rotation[:, 0])
            target_pos = target.value[:,0]
            target_rot = target.rotation[:,0]

            random_dir = np.multiply(np.random.sample(3) - [2.0, 2.5, 0.5], [0.2, 0.2, 0.01])
            ee_pos = target_pos + random_dir
            ee_rot = np.array([target_rot[0] - np.pi/2, 0, 0])
            ik_arm_poses = robot_body.get_ik_from_pose(ee_pos, ee_rot, "left_arm")
            if not len(ik_arm_poses):
                continue
            arm_pose = baxter_sampling.closest_arm_pose(ik_arm_poses, old_l_arm_pose.flatten()).reshape((7,1))
            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': arm_pose, 'rArmPose': old_r_arm_pose, 'lGripper': gripper_val, 'rGripper': gripper_val, 'value': old_pose})

        elif act.name == 'put_into_washer':
            target = act.params[2]
            target_body = act.params[1].openrave_body
            target_body.set_pose(target.value[:, 0], target.rotation[:, 0])
            target_pos = target.value[:,0]
            target_rot = target.rotation[:,0]

            random_dir = np.multiply(np.random.sample(3) - [2.0, 2.5, 0.5], [0.2, 0.2, 0.01])
            ee_pos = target_pos + random_dir
            ee_rot = np.array([target_rot[0] - np.pi/2, 0, 0])
            ik_arm_poses = robot_body.get_ik_from_pose(ee_pos, ee_rot, "left_arm")
            if not len(ik_arm_poses):
                continue
            arm_pose = baxter_sampling.closest_arm_pose(ik_arm_poses, old_l_arm_pose.flatten()).reshape((7,1))
            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': arm_pose, 'rArmPose': old_r_arm_pose, 'lGripper': gripper_val, 'rGripper': gripper_val, 'value': old_pose})

        elif act.name == 'push_door':
            target = act.params[4]
            target_body = act.params[1].openrave_body
            target_body.set_pose(target.value[:, 0], target.rotation[:, 0])
            target_body.set_dof({'door': target.door[:, 0]})
            target_pos = target_body.env_body.GetLink('washer_door').GetTransform()[:3,3]

            random_dir = np.multiply(np.random.sample(3) - [0.5, 1, -4.0], [0.01, 0.05, 0.1])
            ee_pos = target_pos + random_dir
            ee_rot = np.array([np.pi/2, 0, 0])
            ik_arm_poses = robot_body.get_ik_from_pose(ee_pos, ee_rot, "left_arm")
            if not len(ik_arm_poses):
                continue
            arm_pose = baxter_sampling.closest_arm_pose(ik_arm_poses, old_l_arm_pose.flatten()).reshape((7,1))

            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': arm_pose, 'rArmPose': old_r_arm_pose, 'lGripper': gripper_val, 'rGripper': gripper_val, 'value': old_pose})

        elif next_act != None and next_act.name == 'cloth_grasp':
            target = next_act.params[2]
            target_pos = target.value[:, 0]
            # if target pose is not initialized, all entry should be 0
            if np.allclose(target_pos, 0):
                target_pos = next_act.params[1].pose[:, start_ts]
                target.value = target_pos.reshape((3,1))
                target.rotation = next_act.params[1].rotation[:, start_ts].reshape((3,1))
                target._free_attrs['value'][:] = 0
                target._free_attrs['rotation'][:] = 0

            # old_pose = next_act.params[3].value[:,0]
            # robot_body.set_pose([0, 0, old_pose[0]])

            random_dir = np.multiply(np.random.sample(3) - [0.5,0.5,-0.05], [0.01, 0.01, 0.3])
            ee_left = target_pos + random_dir

            l_arm_pose = robot_body.get_ik_from_pose(ee_left, DOWN_ROT, "left_arm")
            if not len(l_arm_pose):
                continue
            l_arm_pose = baxter_sampling.closest_arm_pose(l_arm_pose, old_l_arm_pose.flatten()).reshape((7,1))

            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': l_arm_pose, 'rArmPose': old_r_arm_pose, 'lGripper': np.array([[baxter_constants.GRIPPER_OPEN_VALUE]]), 'rGripper': np.array([[baxter_constants.GRIPPER_OPEN_VALUE]]), 'value': old_pose})

        elif next_act != None and next_act.name == 'cloth_grasp_right':
            target = next_act.params[2]
            target_pos = target.value[:, 0]
            # if target pose is not initialized, all entry should be 0
            if np.allclose(target_pos, 0):
                target_pos = next_act.params[1].pose[:, start_ts]
                target.value = target_pos.reshape((3,1))
                target.rotation = next_act.params[1].rotation[:, start_ts].reshape((3,1))
                target._free_attrs['value'][:] = 0
                target._free_attrs['rotation'][:] = 0

            # old_pose = next_act.params[3].value[:,0]
            # robot_body.set_pose([0, 0, old_pose[0]])

            random_dir = np.multiply(np.random.sample(3) - [0.5,0.5,-0.05], [0.01, 0.01, 0.3])
            ee_right = target_pos + random_dir

            r_arm_pose = robot_body.get_ik_from_pose(ee_right, DOWN_ROT, "right_arm")
            if not len(r_arm_pose):
                continue
            r_arm_pose = baxter_sampling.closest_arm_pose(r_arm_pose, old_r_arm_pose.flatten()).reshape((7,1))

            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': old_l_arm_pose, 'rArmPose': r_arm_pose, 'lGripper': np.array([[baxter_constants.GRIPPER_OPEN_VALUE]]), 'rGripper': np.array([[baxter_constants.GRIPPER_OPEN_VALUE]]), 'value': old_pose})

        elif next_act != None and (next_act.name == 'cloth_putdown' or next_act.name == 'cloth_putdown_in_region_left'):
            target = next_act.params[2]
            target_pos = target.value[:, 0]
            # if target pose is not initialized, all entry should be 0
            if np.allclose(target_pos, 0):
                target_pos = next_act.params[1].pose[:, start_ts]
                target.value = target_pos.reshape((3,1))
                target.rotation = next_act.params[1].rotation[:, start_ts].reshape((3,1))
                target._free_attrs['value'][:] = 0
                target._free_attrs['rotation'][:] = 0

            # old_pose = next_act.params[3].value[:,0]
            # robot_body.set_pose([0, 0, old_pose[0]])

            random_dir = np.multiply(np.random.sample(3) - [0.5,0.5,-1.0], [0.01, 0.01, 0.15])
            ee_left = target_pos + random_dir

            l_arm_pose = robot_body.get_ik_from_pose(ee_left, DOWN_ROT, "left_arm")
            if not len(l_arm_pose):
                continue
            l_arm_pose = baxter_sampling.closest_arm_pose(l_arm_pose, old_l_arm_pose.flatten()).reshape((7,1))

            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': l_arm_pose, 'rArmPose': old_r_arm_pose, 'lGripper': np.array([[baxter_constants.GRIPPER_CLOSE_VALUE]]), 'rGripper': np.array([[baxter_constants.GRIPPER_CLOSE_VALUE]]), 'value': old_pose})

        elif next_act != None and (next_act.name == 'cloth_putdown_right' and next_act.name == 'cloth_putdown_in_region_right'):
            target = next_act.params[2]
            target_pos = target.value[:, 0]
            # if target pose is not initialized, all entry should be 0
            if np.allclose(target_pos, 0):
                target_pos = next_act.params[1].pose[:, start_ts]
                target.value = target_pos.reshape((3,1))
                target.rotation = next_act.params[1].rotation[:, start_ts].reshape((3,1))
                target._free_attrs['value'][:] = 0
                target._free_attrs['rotation'][:] = 0

            # old_pose = next_act.params[3].value[:,0]
            # robot_body.set_pose([0, 0, old_pose[0]])

            random_dir = np.multiply(np.random.sample(3) - [0.5,0.5,-1.0], [0.01, 0.01, 0.15])
            ee_right = target_pos + random_dir

            r_arm_pose = robot_body.get_ik_from_pose(ee_right, DOWN_ROT, "right_arm")
            if not len(r_arm_pose):
                continue
            r_arm_pose = baxter_sampling.closest_arm_pose(r_arm_pose, old_r_arm_pose.flatten()).reshape((7,1))

            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': old_l_arm_pose, 'rArmPose': r_arm_pose, 'lGripper': np.array([[baxter_constants.GRIPPER_CLOSE_VALUE]]), 'rGripper': np.array([[baxter_constants.GRIPPER_CLOSE_VALUE]]), 'value': old_pose})

        elif next_act != None and next_act.name == 'put_into_basket':
            target = next_act.params[4]
            target_body = next_act.params[2].openrave_body
            target_body.set_pose(target.value[:, 0], target.rotation[:, 0])
            target_pos = target.value[:,0]

            random_dir = np.multiply(np.random.sample(3) - [0.5,0.5,-2.0], [0.05, 0.05, 0.1])
            ee_pos = target_pos + random_dir
            ik_arm_poses = robot_body.get_ik_from_pose(ee_pos, DOWN_ROT, "left_arm")
            if not len(ik_arm_poses):
                continue
            arm_pose = baxter_sampling.closest_arm_pose(ik_arm_poses, old_l_arm_pose.flatten()).reshape((7,1))

            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': arm_pose, 'rArmPose': old_r_arm_pose, 'lGripper': gripper_val, 'rGripper': gripper_val, 'value': old_pose})

        elif act.name == 'basket_grasp' or act.name == 'basket_putdown' or act.name == 'basket_grasp_with_cloth' or act.name == 'basket_putdown_with_cloth':
            target = act.params[2]
            target_rot = target.rotation[0, 0]
            handle_dist = baxter_constants.BASKET_OFFSET
            offset = np.array([handle_dist*np.cos(target_rot), handle_dist*np.sin(target_rot), 0])

            act.params[1].openrave_body.set_pose(target.value[:, 0], target.rotation[:, 0])

            random_dir = np.multiply(np.random.sample(3) - [0.5,0.5,-1.0], [0.1, 0.1, 0.1])
            ee_left = target.value[:, 0] + offset + random_dir
            ee_right = target.value[:, 0] - offset + random_dir

            l_arm_pose = robot_body.get_ik_from_pose(ee_left, [target_rot-np.pi/2, np.pi/2, 0], "left_arm")
            r_arm_pose = robot_body.get_ik_from_pose(ee_right, [target_rot-np.pi/2, np.pi/2, 0], "right_arm")
            if not len(l_arm_pose) or not len(r_arm_pose):
                continue
            l_arm_pose = baxter_sampling.closest_arm_pose(l_arm_pose, old_l_arm_pose.flatten()).reshape((7,1))
            r_arm_pose = baxter_sampling.closest_arm_pose(r_arm_pose, old_r_arm_pose.flatten()).reshape((7,1))
            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': l_arm_pose, 'rArmPose': r_arm_pose, 'lGripper': gripper_val, 'rGripper': gripper_val, 'value': old_pose})

        elif act.name == 'cloth_grasp' or act.name == 'cloth_putdown' or act.name == 'cloth_putdown_in_region_left':
            target = act.params[2]
            target_body = act.params[1].openrave_body
            target_body.set_pose(target.value[:, 0], target.rotation[:, 0])
            target_pos = target.value[:, 0]
            random_dir = np.multiply(np.random.sample(3) - [0.5,1.0,-1.5], [0.01, 0.01, 0.1])
            ee_left = target_pos + random_dir

            l_arm_pose = robot_body.get_ik_from_pose(ee_left, DOWN_ROT, "left_arm")
            if not len(l_arm_pose):
                continue
            l_arm_pose = baxter_sampling.closest_arm_pose(l_arm_pose, old_l_arm_pose.flatten()).reshape((7,1))
            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': l_arm_pose, 'rArmPose': old_r_arm_pose, 'lGripper': gripper_val, 'rGripper': gripper_val, 'value': old_pose})

        elif act.name == 'cloth_grasp_right' or act.name == 'cloth_putdown_right' or act.name == 'cloth_putdown_in_region_right':
            target = act.params[2]
            target_body = act.params[1].openrave_body
            target_body.set_pose(target.value[:, 0], target.rotation[:, 0])
            target_pos = target.value[:, 0]
            random_dir = np.multiply(np.random.sample(3) - [0.5,1.0,-1.5], [0.01, 0.01, 0.1])
            ee_right = target_pos + random_dir

            r_arm_pose = robot_body.get_ik_from_pose(ee_right, DOWN_ROT, "left_arm")
            if not len(r_arm_pose):
                continue
            r_arm_pose = baxter_sampling.closest_arm_pose(r_arm_pose, old_r_arm_pose.flatten()).reshape((7,1))
            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': old_l_arm_pose, 'rArmPose': r_arm_pose, 'lGripper': gripper_val, 'rGripper': gripper_val, 'value': old_pose})

        elif act.name == 'put_into_basket':
            target = act.params[4]
            target_body = act.params[2].openrave_body
            target_body.set_pose(target.value[:, 0], target.rotation[:, 0])
            target_pos = target.value[:,0]

            random_dir = np.multiply(np.random.sample(3) - [0.5,-1.5,-2.5], [0.1,0.1,0.1])
            ee_pos = target_pos + random_dir
            ik_arm_poses = robot_body.get_ik_from_pose(ee_pos, DOWN_ROT, "left_arm")
            if not len(ik_arm_poses):
                continue
            arm_pose = baxter_sampling.closest_arm_pose(ik_arm_poses, old_l_arm_pose.flatten()).reshape((7,1))

            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': arm_pose, 'rArmPose': old_r_arm_pose, 'lGripper': gripper_val, 'rGripper': gripper_val, 'value': old_pose})

        elif next_act != None and next_act.name == "open_door":
            target = next_act.params[-2]
            target_body = next_act.params[1].openrave_body
            target_body.set_pose(target.value[:, 0], target.rotation[:, 0])
            target_body.set_dof({'door': target.door[:, 0]})
            target_pos = target_body.env_body.GetLink("washer_handle").GetTransform()[:3,3]

            random_dir = np.multiply(np.random.sample(3) - [.5, 1, 0], [.005, .2, .1])
            ee_pos = target_pos + random_dir
            ee_rot = np.array([np.pi/2, 0, 0])
            ik_arm_poses = robot_body.get_ik_from_pose(ee_pos, ee_rot, "left_arm")
            if not len(ik_arm_poses):
                continue
            arm_pose = baxter_sampling.closest_arm_pose(ik_arm_poses, old_l_arm_pose.flatten()).reshape((7,1))

            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': arm_pose, 'rArmPose': old_r_arm_pose, 'lGripper': gripper_val, 'rGripper': gripper_val, 'value': old_pose})

        elif next_act != None and next_act.name == "close_door":
            target = next_act.params[-2]
            target_body = next_act.params[1].openrave_body
            target_body.set_pose(target.value[:, 0], target.rotation[:, 0])
            target_body.set_dof({'door': target.door[:, 0]})
            target_pos = target_body.env_body.GetLink("washer_handle").GetTransform()[:3,3]

            random_dir = np.multiply(np.random.sample(3) - [.5, 3, 0.5], [.025, .1, .2])
            ee_pos = target_pos + random_dir
            ee_rot = np.array([np.pi/2, 0, 0])
            ik_arm_poses = robot_body.get_ik_from_pose(ee_pos, ee_rot, "left_arm")
            if not len(ik_arm_poses):
                continue
            arm_pose = baxter_sampling.closest_arm_pose(ik_arm_poses, old_l_arm_pose.flatten()).reshape((7,1))

            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': arm_pose, 'rArmPose': old_r_arm_pose, 'lGripper': gripper_val, 'rGripper': gripper_val, 'value': old_pose})

        elif act.name == "open_door":
            target = act.params[-1]
            target_body = act.params[1].openrave_body
            target_body.set_pose(target.value[:, 0], target.rotation[:, 0])
            target_body.set_dof({'door': target.door[:, 0]})
            target_pos = target_body.env_body.GetLink("washer_handle").GetTransform()[:3,3]

            random_dir = np.multiply(np.random.sample(3) - [.5, 3, 0.5], [.025, .1, .2])
            ee_pos = target_pos + random_dir
            ee_rot = np.array([np.pi/2, 0, 0])
            ik_arm_poses = robot_body.get_ik_from_pose(ee_pos, ee_rot, "left_arm")
            if not len(ik_arm_poses):
                continue
            arm_pose = baxter_sampling.closest_arm_pose(ik_arm_poses, old_l_arm_pose.flatten()).reshape((7,1))
            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': arm_pose, 'rArmPose': old_r_arm_pose, 'lGripper': gripper_val, 'rGripper': gripper_val, 'value': old_pose})

        elif act.name == "close_door":
            target = act.params[-1]
            target_body = act.params[1].openrave_body
            target_body.set_pose(target.value[:, 0], target.rotation[:, 0])
            target_body.set_dof({'door': target.door[:, 0]})
            target_pos = target_body.env_body.GetLink("washer_handle").GetTransform()[:3,3]

            random_dir = np.multiply(np.random.sample(3) - [.5, 1, 0], [.005, .2, .1])
            ee_pos = target_pos + random_dir
            ee_rot = np.array([np.pi/2, 0, 0])
            ik_arm_poses = robot_body.get_ik_from_pose(ee_pos, ee_rot, "left_arm")
            if not len(ik_arm_poses):
                continue
            arm_pose = baxter_sampling.closest_arm_pose(ik_arm_poses, old_l_arm_pose.flatten()).reshape((7,1))
            # TODO once we have the rotor_base we should resample pose
            robot_pose.append({'lArmPose': arm_pose, 'rArmPose': old_r_arm_pose, 'lGripper': gripper_val, 'rGripper': gripper_val, 'value': old_pose})

        elif act.name == "rotate":
            target_rot = act.params[3]
            init_pos = act.params[1]
            robot_pose.append({'lArmPose': init_pos.lArmPose.copy(), 'rArmPose': init_pos.rArmPose.copy(), 'lGripper': init_pos.lGripper.copy(), 'rGripper': init_pos.rGripper.copy(), 'value': target_rot.value.copy()})

        elif next_act != None and next_act.name == "rotate":
            init_pos = act.params[1]
            robot_body.set_dof({'lArmPose': init_pos.lArmPose[:, 0], 'rArmPose': init_pos.rArmPose[:, 0]})
            l_random_dir = np.multiply(np.random.sample(3) - [0.5, 0.5, 0], [0.2, 0.2, 0.2])
            l_ee_pos = robot_body.env_body.GetLink('left_gripper').GetTransform()[:3, 3] + l_random_dir
            l_arm_poses = robot_body.get_ik_from_pose(l_ee_pos, [0, np.pi/2, 0], "left_arm")
            r_random_dir = np.multiply(np.random.sample(3) - [0.5, 0.5, 0], [0.2, 0.2, 0.2])
            r_ee_pos = robot_body.env_body.GetLink('right_gripper').GetTransform()[:3, 3] + r_random_dir
            r_arm_poses = robot_body.get_ik_from_pose(r_ee_pos, [0, np.pi/2, 0], "right_arm")
            if not len(l_arm_poses) or not len(r_arm_poses):
                continue
            robot_pose.append({'lArmPose': l_arm_poses[0], 'rArmPose': r_arm_poses[0], 'lGripper': init_pos.lGripper.copy(), 'rGripper': init_pos.rGripper.copy(), 'value': init_pos.value.copy()})

        else:
            import ipdb; ipdb.set_trace()
            raise NotImplementedError
    if not robot_pose:
        print("Unable to find IK")
        # import ipdb; ipdb.set_trace()

    return robot_pose
