import xml.etree.ElementTree as xml

from mujoco_py import mjcore

def generate_xml(plan, color_map):
    '''
        Search a plan for cloths, tables, and baskets to create an XML in MJCF format
    '''
    base_xml = xml.parse(BASE_POS_XML)
    root = base_xml.getroot()
    worldbody = root.find('worldbody')
    active_ts = (0, plan.horizon)
    params = list(plan.params.values())
    contacts = root.find('contact')
    equality = root.find('equality')

    cur_eq_ind = 0
    equal_active_inds = {}
    for param in params:
        if param.is_symbol(): continue
        if param._type == 'Cloth':
            height = param.geom.height
            radius = param.geom.radius * 2.5
            x, y, z = param.pose[:, active_ts[0]]
            color = color_map[param.name] if param.name in color_map else "1 1 1 1"
            cloth_body = xml.SubElement(worldbody, 'body', {'name': param.name, 'pos': "{} {} {}".format(x,y,z+MUJOCO_MODEL_Z_OFFSET), 'euler': "0 0 0"})
            # cloth_geom = xml.SubElement(cloth_body, 'geom', {'name':param.name, 'type':'cylinder', 'size':"{} {}".format(radius, height), 'rgba':"0 0 1 1", 'friction':'1 1 1'})
            cloth_geom = xml.SubElement(cloth_body, 'geom', {'name': param.name, 'type':'sphere', 'size':"{}".format(radius), 'rgba':color, 'friction':'1 1 1'})
            cloth_intertial = xml.SubElement(cloth_body, 'inertial', {'pos':'0 0 0', 'quat':'0 0 0 1', 'mass':'0.1', 'diaginertia': '0.01 0.01 0.01'})
            xml.SubElement(equality, 'connect', {'body1': param.name, 'body2': 'left_gripper_l_finger_tip', 'anchor': "0 0 0", 'active':'false'})
            equal_active_inds[(param.name, 'left_gripper')] = cur_eq_ind
            cur_eq_ind += 1
            # Exclude collisions between the left hand and the cloth to help prevent exceeding the contact limit
            xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_wrist'})
            xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_hand'})
            xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_base'})
            xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_l_finger_tip'})
            xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_l_finger'})
            xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_r_finger_tip'})
            xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_r_finger'})
            xml.SubElement(contacts, 'exclude', {'body1': param.name, 'body2': 'basket'})
        elif param._type == 'Obstacle':
            length = param.geom.dim[0]
            width = param.geom.dim[1]
            thickness = param.geom.dim[2]
            x, y, z = param.pose[:, active_ts[0]]
            table_body = xml.SubElement(worldbody, 'body', {'name': param.name, 'pos': "{} {} {}".format(x, y, z+MUJOCO_MODEL_Z_OFFSET), 'euler': "0 0 0"})
            table_geom = xml.SubElement(table_body, 'geom', {'name':param.name, 'type':'box', 'size':"{} {} {}".format(length, width, thickness)})
        elif param._type == 'Basket':
            x, y, z = param.pose[:, active_ts[0]]
            yaw, pitch, roll = param.rotation[:, active_ts[0]]
            basket_body = xml.SubElement(worldbody, 'body', {'name':param.name, 'pos':"{} {} {}".format(x, y, z+MUJOCO_MODEL_Z_OFFSET), 'euler':'{} {} {}'.format(pitch, roll, yaw)})
            basket_intertial = xml.SubElement(basket_body, 'inertial', {'pos':"0 0 0", 'mass':"0.1", 'diaginertia':"2 1 1"})
            basket_geom = xml.SubElement(basket_body, 'geom', {'name':param.name, 'type':'mesh', 'mesh': "laundry_basket"})

            basket_left_handle = xml.SubElement(basket_body, 'body', {'name': 'basket_left_handle', 'pos':"{} {} {}".format(0.317, 0, 0), 'euler':'0 0 0'})
            basket_geom = xml.SubElement(basket_left_handle, 'geom', {'type':'sphere', 'size': '0.01'})
            xml.SubElement(contacts, 'exclude', {'body1': 'basket_left_handle', 'body2': 'basket'})
            xml.SubElement(contacts, 'exclude', {'body1': 'basket_left_handle', 'body2': 'left_gripper_l_finger_tip'})
            xml.SubElement(contacts, 'exclude', {'body1': 'basket_left_handle', 'body2': 'left_gripper_r_finger_tip'})
            xml.SubElement(equality, 'connect', {'body1': 'basket_left_handle', 'body2': 'basket', 'anchor': "0 0.317 0", 'active':'true'})
            xml.SubElement(equality, 'connect', {'body1': 'basket_left_handle', 'body2': 'left_gripper_r_finger_tip', 'anchor': "0 0 0", 'active':'false'})
            equal_active_inds[('basket', 'left_handle')] = cur_eq_ind
            cur_eq_ind += 1
            equal_active_inds[('left_handle', 'left_gripper')] = cur_eq_ind
            cur_eq_ind += 1

            basket_right_handle = xml.SubElement(basket_body, 'body', {'name': 'basket_right_handle', 'pos':"{} {} {}".format(-0.317, 0, 0), 'euler':'0 0 0'})
            basket_geom = xml.SubElement(basket_right_handle, 'geom', {'type':'sphere', 'size': '0.01'})
            xml.SubElement(contacts, 'exclude', {'body1': 'basket_right_handle', 'body2': 'basket'})
            xml.SubElement(contacts, 'exclude', {'body1': 'basket_right_handle', 'body2': 'right_gripper_l_finger_tip'})
            xml.SubElement(contacts, 'exclude', {'body1': 'basket_right_handle', 'body2': 'right_gripper_r_finger_tip'})
            xml.SubElement(equality, 'connect', {'body1': 'basket_right_handle', 'body2': 'basket', 'anchor': "0 -0.317 0", 'active':'true'})
            xml.SubElement(equality, 'connect', {'body1': 'basket_right_handle', 'body2': 'right_gripper_l_finger_tip', 'anchor': "0 0 0", 'active':'false'})
            equal_active_inds[('basket', 'right_handle')] = cur_eq_ind
            cur_eq_ind += 1
            equal_active_inds[('right_handle', 'right_gripper')] = cur_eq_ind
            cur_eq_ind += 1
    base_xml.write(ENV_XML)
    return equal_active_inds


def setup_mjc_model(plan, view=False, color_map={}):
    '''
        Create the Mujoco model and intiialize the viewer if desired
    '''
    eq_active_inds = generate_xml(plan, color_map)
    model = mjcore.MjModel(ENV_XML)

    viewer = None
    if view:
        viewer = mjviewer.MjViewer()
        viewer.start()
        viewer.set_model(model)
        viewer.cam.distance = 3
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -37.5
        viewer.loop_once()

    return model, viewer, eq_active_inds
