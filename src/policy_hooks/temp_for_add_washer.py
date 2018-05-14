
            if param._type == 'Cloth':
                height = param.geom.height
                radius = param.geom.radius * 2.5
                x, y, z = param.pose[:, active_ts[0]]
                color = self.color_maps[cond][param.name]
                washer_body = xml.SubElement(worldbody, 'body', {'name': param.name, 'pos': "{} {} {}".format(x,y,z+MUJOCO_MODEL_Z_OFFSET), 'euler': "0 0 0"})
                washer_back = xml.SubElement(washer_body, 'body', {'name': param.name, 'pos': "0 -0.2 0", 'euler': "0 0 0"})
                washer_back_geom = xml.SubElement(washer_back, 'geom', {'name':param.name, 'type':'box', 'size':"0.3 0.01 0.3"})
                washer_left = xml.SubElement(washer_body, 'body', {'name': param.name, 'pos': "-0.3 0.1 0", 'euler': "0 0 0"})
                washer_left_geom = xml.SubElement(washer_left, 'geom', {'name':param.name, 'type':'box', 'size':"0.01 0.3 0.3"})
                washer_right = xml.SubElement(washer_body, 'body', {'name': param.name, 'pos': "0.3 0.1 0", 'euler': "0 0 0"})
                washer_right_geom = xml.SubElement(washer_right, 'geom', {'name':param.name, 'type':'box', 'size':"0.3 0.01 0.3"})
                washer_bottom = xml.SubElement(washer_body, 'body', {'name': param.name, 'pos': "0 0.1 -0.3", 'euler': "0 0 0"})
                washer_bottom_geom = xml.SubElement(washer_bottom, 'geom', {'name':param.name, 'type':'box', 'size':"0.3 0.3 0.01"})
                washer_top = xml.SubElement(washer_body, 'body', {'name': param.name, 'pos': "0 0.1 0.3", 'euler': "0 0 0"})
                washer_top_geom = xml.SubElement(washer_top, 'geom', {'name':param.name, 'type':'box', 'size':"0.3 0.3 0.01"})

                washer_door = xml.SubElement(washer_body, 'body', {'name': param.name, 'pos': "0 0.4 0", 'euler': "1.57 0 0"})
                washer_door_geom = xml.SubElement(washer_door, 'geom', {'name': param.name, 'type':'cylinder', 'size':"0.3 0.01", 'rgba':'1 1 1'})
                washer_handle = xml.SubElement(washer_body, 'body', {'name': param.name, 'pos': "0 0 0.4", 'euler': "0 0 0"})
                washer_door_geom = xml.SubElement(washer_handle, 'geom', {'name': param.name, 'type':'cylinder', 'size':"0.01 0.1", 'rgba':'1 1 1'})
                