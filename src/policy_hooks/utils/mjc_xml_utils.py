import xml.etree.ElementTree as xml

MUJOCO_MODEL_Z_OFFSET = -0.706

def get_param_xml(param):
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
            xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'left_wrist'}),
            xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'left_hand'}),
            xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_base'}),
            xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_l_finger_tip'}),
            xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_l_finger'}),
            xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_r_finger_tip'}),
            xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'left_gripper_r_finger'}),
            xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'right_wrist'}),
            xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'right_hand'}),
            xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'right_gripper_base'}),
            xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'right_gripper_l_finger_tip'}),
            xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'right_gripper_l_finger'}),
            xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'right_gripper_r_finger_tip'}),
            xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'right_gripper_r_finger'}),
            xml.Element(contacts, 'exclude', {'body1': param.name, 'body2': 'basket'}),
        ]

        return param.name, cloth_body, {'contacts': contacts}

    elif param._type == 'Obstacle': 
        length = param.geom.dim[0]
        width = param.geom.dim[1]
        thickness = param.geom.dim[2]
        x, y, z = param.pose[:, active_ts[0]]
        table_body = xml.Element('body', {'name': param.name, 
                                          'pos': "{} {} {}".format(x, y, z+MUJOCO_MODEL_Z_OFFSET), 
                                          'euler': "0 0 0"})
        table_geom = xml.SubElement(table_body, 'geom', {'name':param.name, 
                                                         'type':'box', 
                                                         'size':"{} {} {}".format(length, width, thickness)})
        return param.name, table_body, {'contacts': []}

    elif param._type == 'Basket':
        x, y, z = param.pose[:, active_ts[0]]
        yaw, pitch, roll = param.rotation[:, active_ts[0]]
        basket_body = xml.Element('body', {'name':param.name, 
                                  'pos':"{} {} {}".format(x, y, z+MUJOCO_MODEL_Z_OFFSET), 
                                  'euler':'{} {} {}'.format(roll, pitch, yaw), 
                                  'mass': "1"})
        # basket_intertial = xml.SubElement(basket_body, 'inertial', {'pos':"0 0 0", 'mass':"0.1", 'diaginertia':"2 1 1"})
        basket_geom = xml.SubElement(basket_body, 'geom', {'name':param.name, 
                                                           'type':'mesh', 
                                                           'mesh': "laundry_basket"})
        return param.name, basket_body, {'contacts': []}


def get_deformable_cloth(width, length, pos=(1.,0.,1.), material='matcarpet', label=""):
    spacing = 0.05
    body =  '''
                <body name="B0_0" pos="{0} {1} {2}">
                    <freejoint />
                    <composite type="cloth" count="{3} {4} 1" spacing="{5}" flatinertia="0.01">
                        <joint kind="main" damping="0.001"/>
                        <skin material="cloth_1" texcoord="true" inflate="0.005" subgrid="3" />
                        <geom type="capsule" size="0.015 0.01" rgba=".8 .2 .1 1" />
                    </composite>
                </body>\n
                '''.format(pos[0], pos[1], pos[2], length, width, spacing)

    texture = '<texture name="cloth_1" type="2d" file="cloth_1.png" />'
    xml_texture = xml.fromstring(texture)

    material = '<material name="cloth_1" texture="cloth_1" shininess="0.25" />'
    xml_material = xml.fromstring(material)

    # corner_1 = '<site name="{0}_corner_1" pos="1. 0 1."/>\n'.format(label)
    # corner_2 = '<site name="{0}_corner_2" pos="{1} 0 1."/>\n'.format(label, 1+spacing*length)
    # corner_3 = '<site name="{0}_corner_3" pos="{1} {2} 1."/>\n'.format(label, 1+spacing*length, spacing*width)
    # corner_4 = '<site name="{0}_corner_4" pos="1. {1} 1."/>\n'.format(label, spacing*width)
    # center = '<site name="{0}_center" pos="{1} {2} 1."/>\n'.format(label, 1+spacing*length/2, spacing*width/2)

    xml_body = xml.fromstring(body)
    # xml_corner_1 = xml.fromstring(corner_1)
    # xml_corner_2 = xml.fromstring(corner_2)
    # xml_corner_3 = xml.fromstring(corner_3)
    # xml_corner_4 = xml.fromstring(corner_4)
    # xml_center = xml.fromstring(center)
    # sites = [xml_corner_1, xml_corner_2, xml_corner_3, xml_corner_4, xml_center]


    return 'B0_0', xml_body, {'assets': [xml_texture, xml_material]}


def generate_xml(base_file, target_file, items):
    base_xml = xml.parse(base_file)
    root = base_xml.getroot()
    worldbody = root.find('worldbody')
    contacts = root.find('contact')
    assets = root.find('asset')

    for name, item_body, tag_dict in items:
        worldbody.append(item_body)
        if 'contacts' in tag_dict:
            for contact in tag_dict['contacts']:
                contacts.append(contact)
        if 'assets' in tag_dict:
            for asset in tag_dict['assets']:
                assets.append(asset)

    base_xml.write(target_file)
