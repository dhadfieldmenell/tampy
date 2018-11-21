import xml.etree.ElementTree as xml

MUJOCO_MODEL_Z_OFFSET = -0.706

def get_xml(self, param):
	if param._type == 'Cloth':
        height = param.geom.height
        radius = param.geom.radius
        x, y, z = param.pose[:, 0]
        cloth_body = xml.Element('body', {'name': param.name, 'pos': "{} {} {}".format(x,y,z+MUJOCO_MODEL_Z_OFFSET), 'euler': "0 0 0"})
        # cloth_geom = xml.SubElement(cloth_body, 'geom', {'name':param.name, 'type':'cylinder', 'size':"{} {}".format(radius, height), 'rgba':"0 0 1 1", 'friction':'1 1 1'})
        cloth_geom = xml.SubElement(cloth_body, 'geom', {'name': param.name, 'type':'sphere', 'size':"{}".format(radius), 'rgba':"0 0 1 1", 'friction':'1 1 1'})
        cloth_intertial = xml.SubElement(cloth_body, 'inertial', {'pos':'0 0 0', 'quat':'0 0 0 1', 'mass':'0.1', 'diaginertia': '0.01 0.01 0.01'})
        # Exclude collisions between the left hand and the cloth to help prevent exceeding the contact limit
        contacts = [
	        xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'left_wrist'})
	        xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'left_hand'})
	        xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_base'})
	        xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_l_finger_tip'})
	        xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_l_finger'})
	        xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_r_finger_tip'})
	        xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_r_finger'})
	        xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'basket'})
        ]

        return cloth_body, {'contacts': contacts}

    elif param._type == 'Obstacle': 
        length = param.geom.dim[0]
        width = param.geom.dim[1]
        thickness = param.geom.dim[2]
        x, y, z = param.pose[:, active_ts[0]]
        table_body = xml.Element('body', {'name': param.name, 'pos': "{} {} {}".format(x, y, z+MUJOCO_MODEL_Z_OFFSET), 'euler': "0 0 0"})
        table_geom = xml.SubElement(table_body, 'geom', {'name':param.name, 'type':'box', 'size':"{} {} {}".format(length, width, thickness)})
        return table_body, {'contacts': []}

    elif param._type == 'Basket':
        x, y, z = param.pose[:, active_ts[0]]
        yaw, pitch, roll = param.rotation[:, active_ts[0]]
        basket_body = xml.Element('body', {'name':param.name, 'pos':"{} {} {}".format(x, y, z+MUJOCO_MODEL_Z_OFFSET), 'euler':'{} {} {}'.format(roll, pitch, yaw)})
        basket_intertial = xml.SubElement(basket_body, 'inertial', {'pos':"0 0 0", 'mass':"0.1", 'diaginertia':"2 1 1"})
        basket_geom = xml.SubElement(basket_body, 'geom', {'name':param.name, 'type':'mesh', 'mesh': "laundry_basket"})
        return basket_body, {'contacts': []}


def get_deformable_cloth(self, name, width, length, material='matcarpet'):
    body = '''
            <body name={0} pos="0 0 1">
                <freejoint/>
                <composite type="cloth" count="{1} {2} 1" spacing="0.05" flatinertia="0.01">
                    <joint kind="main" damping="0.001"/>
                    <skin material={3} texcoord="true" inflate="0.005" subgrid="2"/>
                    <geom type="capsule" size="0.015 0.01" rgba=".8 .2 .1 1"/>
                </composite>
            </body> 
            '''.format(name, width, length, material)
    return body, {'contacts': []}
