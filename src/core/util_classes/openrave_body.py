import numpy as np
from math import cos, sin, atan2

import pybullet as P

import core.util_classes.common_constants as const
import core.util_classes.transform_utils as T
from core.util_classes.robots import Robot, PR2, Baxter, Washer, NAMO
from core.util_classes.items import Item, Box, Can, BlueCan, RedCan, Circle, \
                                    BlueCircle, RedCircle, GreenCircle, Obstacle, \
                                    Wall, Table, Basket, XMLItem, Door

WALL_THICKNESS = 1
CLOSET_POINTS = [[-6.0,-8.0],[-6.0,4.0],[1.9,4.0],[1.9,8.0],[5.0,8.0],[5.0,4.0],[13.0,4.0],[13.0,-8.0],[-6.0,-8.0]]
CLOSET_POINTS = [[-7.0,-9.0],[-7.0,4.0],[1.9,4.0],[1.9,8.0],[5.0,8.0],[5.0,4.0],[14.0,4.0],[14.0,-9.0],[-7.0,-9.0]]
CLOSET_POINTS = [[-7.0,-10.0],[-7.0,4.0],[1.9,4.0],[1.9,8.0],[5.0,8.0],[5.0,4.0],[14.0,4.0],[14.0,-10.0],[-7.0,-10.0]]
CLOSET_POINTS = [[-7.0,-12.0],[-7.0,4.0],[1.9,4.0],[1.9,8.0],[5.0,8.0],[5.0,4.0],[14.0,4.0],[14.0,-12.0],[-7.0,-12.0]]
CLOSET_POINTS = [[-7.0,-10.0],[-7.0,4.0],[1.5,4.0],[1.5,8.0],[5.5,8.0],[5.5,4.0],[14.0,4.0],[14.0,-10.0],[-7.0,-10.0]]
#CLOSET_POINTS = [[-7.0,-10.0],[-7.0,4.0],[1.5,4.0],[5.5,4.0],[14.0,4.0],[14.0,-10.0],[-7.0,-10.0]]

class OpenRAVEBody(object):
    def __init__(self, env, name, geom):
        self.name = name
        self._env = env
        self._geom = geom

        if isinstance(geom, Robot):
            self._add_robot(geom)
        elif isinstance(geom, XMLItem):
            self._add_xml_item(geom)
        elif isinstance(geom, Item):
            self._add_item(geom)
        else:
            raise OpenRAVEException("Geometry not supported for %s for OpenRAVEBody"%geom)

        # self.set_transparency(0.5)

    def delete(self):
        P.removeCollisionShape(self.body_id)

    def isrobot(self):
        return isinstance(self._geom, Robot)

    def set_transparency(self, transparency):
        visual_infos = P.getVisualShapeData(self.body_id)
        for info in visual_infos:
            link_index = info[1]
            link_rgba = info[7]
            P.changeVisualShape(self.body_id, link_index, rgbaColor=list(link_rgba[:3])+[transparency])

    def _add_robot(self, geom):
        if not geom.is_initialized():
            geom.setup(None)
        self.env_body = geom.id
        self.body_id = geom.id

    def _add_xml_item(self, geom):
        if not geom.is_initialized():
            geom.setup(None)
        self.env_body = geom.id
        self.body_id = geom.id

    def _add_item(self, geom):
        try:
            fun_name = "self._add_{}".format(geom._type)
            eval(fun_name)(geom)
        except Exception as e:
            print('Could not add', geom._type, e)
            raise e
            #self._add_obj(geom)

    def _add_circle(self, geom):
        color = [1,0,0]
        if hasattr(geom, "color") and geom.color == 'blue':
            color = [0, 0, 1]
        elif hasattr(geom, "color") and geom.color == 'green':
            color = [0, 1, 0]
        elif hasattr(geom, "color") and geom.color == 'red':
            color = [1, 0, 0]

        self.col_body_id = P.createCollisionShape(shapeType=P.GEOM_CYLINDER, radius=geom.radius, height=2)
        self.body_id = P.createMultiBody(1, self.col_body_id)

    def _add_can(self, geom):
        color = [1,0,0]
        if hasattr(geom, "color") and geom.color == 'blue':
            color = [0, 0, 1]
        elif hasattr(geom, "color") and geom.color == 'green':
            color = [0, 1, 0]
        elif hasattr(geom, "color") and geom.color == 'red':
            color = [1, 0, 0]

        self.col_body_id = P.createCollisionShape(shapeType=P.GEOM_CYLINDER, radius=geom.radius, height=geom.height)
        self.body_id = P.createMultiBody(1, self.col_body_id)

    #def _add_obstacle(self, geom):
    #    obstacles = np.matrix('-0.576036866359447, 0.918128654970760, 1;\
    #                    -0.806451612903226,-1.07017543859649, 1;\
    #                    1.01843317972350,-0.988304093567252, 1;\
    #                    0.640552995391705,0.906432748538011, 1;\
    #                    -0.576036866359447, 0.918128654970760, -1;\
    #                    -0.806451612903226,-1.07017543859649, -1;\
    #                    1.01843317972350,-0.988304093567252, -1;\
    #                    0.640552995391705,0.906432748538011, -1')

    #    body = RaveCreateKinBody(self._env, '')
    #    vertices = np.array(obstacles)
    #    indices = np.array([[0, 1, 2], [2, 3, 0], [4, 5, 6], [6, 7, 4], [0, 4, 5],
    #                        [0, 1, 5], [1, 2, 5], [5, 6, 2], [2, 3, 6], [6, 7, 3],
    #                        [0, 3, 7], [0, 4, 7]])
    #    body.InitFromTrimesh(trimesh=TriMesh(vertices, indices), draw=True)
    #    body.SetName(self.name)
    #    for link in body.GetLinks():
    #        for geom in link.GetGeometries():
    #            geom.SetDiffuseColor((.9, .9, .9))
    #    self.env_body = body
    #    self._env.AddKinBody(body)

    def _add_box(self, geom):
        self.col_body_id = P.createCollisionShape(shapeType=P.GEOM_BOX, halfExtents=geom.dim)
        self.body_id = P.createMultiBody(1, self.col_body_id)

    def _add_sphere(self, geom):
        self.col_body_id = P.createCollisionShape(shapeType=P.GEOM_SPHERE, radius=geom.radius)
        self.body_id = P.createMultiBody(1, self.col_body_id)

    def _add_door(self, geom):
        self.body_id, self.col_body_id = OpenRAVEBody.create_door(self._env, geom.length)

    def _add_wall(self, geom):
        self.body_id = OpenRAVEBody.create_wall(self._env, geom.wall_type)

    #def _add_obj(self, geom):
    #    self.env_body = self._env.ReadKinBodyXMLFile(geom.shape)
    #    self.env_body.SetName(self.name)
    #    self._env.Add(self.env_body)

    #def _add_table(self, geom):
    #    self.env_body = OpenRAVEBody.create_table(self._env, geom)
    #    self.env_body.SetName(self.name)
    #    self._env.Add(self.env_body)

    #def _add_basket(self, geom):
    #    self.env_body = self._env.ReadKinBodyXMLFile(geom.shape)
    #    self.env_body.SetName(self.name)
    #    self._env.Add(self.env_body)

    def set_pose(self, base_pose, rotation = [0, 0, 0]):
        trans = None
        if np.any(np.isnan(base_pose)) or np.any(np.isnan(rotation)):
            return

        if hasattr(self._geom, 'jnt_names') and 'pose' in self._geom.jnt_names:
            dof_map = {'pose': base_pose}
            return self.set_dof(dof_map)

        if isinstance(self._geom, Baxter):
            pos = np.r_[base_pose[:2], 0]
            quat = T.euler_to_quaternion([0, 0, base_pose[2]], order='xyzw')
        elif len(base_pose) == 2:
            base_pose = np.array(base_pose).flatten()
            pos = np.concatenate([base_pose, [0]]).flatten()
            if len(rotation) == 1:
                rotation = [0., 0., rotation[0]]
            # quat = [0, 0, 0, 1]
            quat = T.euler_to_quaternion(rotation, order='xyzw')
        else:
            pos = base_pose
            quat = T.euler_to_quaternion(rotation, order='xyzw')
        P.resetBasePositionAndOrientation(self.body_id, pos, quat)

    def set_dof(self, dof_value_map, debug=False):
        """
            dof_value_map: A dict that maps robot attribute name to a list of corresponding values
        """
        # make sure only sets dof for robot
        # assert isinstance(self._geom, Robot)
        #if not isinstance(self._geom, Robot): return
        if not hasattr(self._geom, 'dof_map'): return

        for key in dof_value_map:
            if key not in self._geom.dof_map:
                if debug: print('Cannot set dof for', key)
                continue

            if type(self._geom.dof_map[key]) is int:
                P.resetJointState(self.body_id, self._geom.dof_map[key], dof_value_map[key])
            else:
                for i, jnt_ind in enumerate(self._geom.dof_map[key]):
                    if type(dof_value_map[key]) is int:
                        val = dof_value_map[key]
                    else:
                        ind = min(i, len(dof_value_map[key])-1)
                        val = dof_value_map[key][ind]
                    P.resetJointState(self.body_id, jnt_ind, val)

    def _set_active_dof_inds(self, inds = None):
        """
        Set active dof index to the one we are interested
        This function is implemented to simplify jacobian calculation in the CollisionPredicate
        inds: Optional list of index specifying dof index we are interested in
        """
        pass

    #@staticmethod
    #def create_cylinder(env, body_name, t, dims, color=[0, 1, 1]):
    #    infocylinder = OpenRAVEBody.create_body_info(GeometryType.Cylinder, dims, color)
    #    if type(env) != Environment:
    #        print("Environment object is not valid")
    #    cylinder = RaveCreateKinBody(env, '')
    #    cylinder.InitFromGeometries([infocylinder])
    #    cylinder.SetName(body_name)
    #    cylinder.SetTransform(t)
    #    return cylinder

    #@staticmethod
    #def create_box(env, name, transform, dims, color=[0,0,1]):
    #    infobox = OpenRAVEBody.create_body_info(KinBody.Link.GeomType.Box, dims, color, 0, True)
    #    box = RaveCreateKinBody(env,'')
    #    box.InitFromGeometries([infobox])
    #    box.SetName(name)
    #    box.SetTransform(transform)
    #    return box

    #@staticmethod
    #def create_sphere(env, name, transform, dims, color=[0,0,1]):
    #    infobox = OpenRAVEBody.create_body_info(GeometryType.Sphere, dims, color)
    #    sphere = RaveCreateKinBody(env,'')
    #    sphere.InitFromGeometries([infobox])
    #    sphere.SetName(name)
    #    sphere.SetTransform(transform)
    #    return sphere

    #@staticmethod
    #def create_body_info(body_type, dims, color, transparency = 0.8, visible = True):
    #    infobox = KinBody.Link.GeometryInfo()
    #    infobox._type = body_type
    #    infobox._vGeomData = dims
    #    infobox._bVisible = True
    #    infobox._fTransparency = transparency
    #    infobox._vDiffuseColor = color
    #    return infobox

    @staticmethod
    def create_door(env, door_len):
        from core.util_classes.namo_grip_predicates import HANDLE_OFFSET
        door_color = [0.5, 0.2, 0.1]
        box_infos = []
        cols = [P.createCollisionShape(shapeType=P.GEOM_CYLINDER, radius=0.05, height=0.1),
                P.createCollisionShape(shapeType=P.GEOM_BOX, halfExtents=[door_len/2.-0.1, 0.1, 0.4]),
                P.createCollisionShape(shapeType=P.GEOM_CYLINDER, radius=0.3, height=0.4),]
        link_pos = [(0, 0, 0), (door_len/2., 0., 0.), (door_len/2., -HANDLE_OFFSET, 0.)]
        door = P.createMultiBody(basePosition=[0,0,0],
                                linkMasses=[1 for _ in cols],
                                linkCollisionShapeIndices=[ind for ind in cols],
                                linkVisualShapeIndices=[-1 for _ in cols],
                                linkPositions=[pos for pos in link_pos],
                                linkOrientations=[[0,0,0,1] for _ in cols],
                                linkInertialFramePositions=[[0,0,0] for _ in cols],
                                linkInertialFrameOrientations=[[0,0,0,1] for _ in cols],
                                linkParentIndices=[0, 1, 2],
                                linkJointTypes=[P.JOINT_REVOLUTE]+[P.JOINT_FIXED for _ in cols[1:]],
                                linkJointAxis=[[0,0,1] for _ in cols]
                               )
        return door, cols

    @staticmethod
    def create_wall(env, wall_type):
        wall_color = [0.5, 0.2, 0.1]
        box_infos = []
        if wall_type == 'closet':
            # wall_endpoints = [[-6.0,-8.0],[-6.0,4.0],[1.9,4.0],[1.9,8.0],[5.0,8.0],[5.0,4.0],[13.0,4.0],[13.0,-8.0],[-6.0,-8.0]]
            wall_endpoints = CLOSET_POINTS
        elif wall_type == 'three_room':
            wall_endpoints = [[-6.0,-8.0],[-6.0,4.0],[-1.5,4.0],
                              [-1.5,2.0],[-1.5,4.0],[6.0,4.0],
                              [6.0,2.0],[6.0,4.0],[13.0,4.0],
                              [13.0,-8.0],[6.0,-8.0],[6.0,-1.5],
                              [6.0,-8.0],[-1.5,-8.0],[-1.5,-1.5],
                              [-1.5,-8.0], [-6.0,-8.0]]
        else:
            raise NotImplemented
        for i, (start, end) in enumerate(zip(wall_endpoints[0:-1], wall_endpoints[1:])):
            dim_x, dim_y = 0, 0
            thickness = WALL_THICKNESS
            if start[0] == end[0]:
                ind_same, ind_diff = 0, 1
                length = abs(start[ind_diff] - end[ind_diff])
                dim_x, dim_y = thickness, length/2 + thickness
            elif start[1] == end[1]:
                ind_same, ind_diff = 1, 0
                length = abs(start[ind_diff] - end[ind_diff])
                dim_x, dim_y = length/2 + thickness, thickness
            else:
                raise NotImplemented('Can only create axis-aligned walls')

            transform = np.eye(4)
            transform[ind_same, 3] = start[ind_same]
            if start[ind_diff] < end[ind_diff]:
                transform[ind_diff, 3] = start[ind_diff] + length/2
            else:
                transform[ind_diff, 3] = end[ind_diff] + length/2
            dims = [dim_x, dim_y, 1]
            box_infos.append((dims, transform[:3,3]))
        cols = [P.createCollisionShape(shapeType=P.GEOM_BOX, halfExtents=h) for h, t in box_infos]
        wall = P.createMultiBody(basePosition=[0,0,0],
                                linkMasses=[1 for _ in cols],
                                linkCollisionShapeIndices=[ind for ind in cols],
                                linkVisualShapeIndices=[-1 for _ in cols],
                                linkPositions=[t[:3] for _, t in box_infos],
                                linkOrientations=[[0,0,0,1] for _, t in box_infos],
                                linkInertialFramePositions=[[0,0,0] for _ in cols],
                                linkInertialFrameOrientations=[[0,0,0,1] for _, t in box_infos],
                                linkParentIndices=[0 for _ in cols],
                                linkJointTypes=[P.JOINT_FIXED for _ in cols],
                                linkJointAxis=[[0,0,1] for _ in cols]
                               )
        return wall


    @staticmethod
    def get_wall_dims(wall_type='closet'):
        if wall_type == 'closet':
            wall_endpoints = CLOSET_POINTS
        elif wall_type == 'three_room':
            wall_endpoints = [[-6.0,-8.0],[-6.0,4.0],[-1.5,4.0],
                              [-1.5,2.0],[-1.5,4.0],[6.0,4.0],
                              [6.0,2.0],[6.0,4.0],[13.0,4.0],
                              [13.0,-8.0],[6.0,-8.0],[6.0,-1.5],
                              [6.0,-8.0],[-1.5,-8.0],[-1.5,-1.5],
                              [-1.5,-8.0], [-6.0,-8.0]]
        else:
            raise NotImplemented

        dims = []
        for i, (start, end) in enumerate(zip(wall_endpoints[0:-1], wall_endpoints[1:])):
            dim_x, dim_y = 0, 0
            thickness = WALL_THICKNESS
            if start[0] == end[0]:
                ind_same, ind_diff = 0, 1
                length = abs(start[ind_diff] - end[ind_diff])
                dim_x, dim_y = thickness, length/2 + thickness
            elif start[1] == end[1]:
                ind_same, ind_diff = 1, 0
                length = abs(start[ind_diff] - end[ind_diff])
                dim_x, dim_y = length/2 + thickness, thickness
            else:
                raise NotImplemented('Can only create axis-aligned walls')

            transform = np.eye(4)
            transform[ind_same, 3] = start[ind_same]
            if start[ind_diff] < end[ind_diff]:
                transform[ind_diff, 3] = start[ind_diff] + length/2
            else:
                transform[ind_diff, 3] = end[ind_diff] + length/2
            dims.append(([dim_x, dim_y, 1], transform))
        return dims


    #@staticmethod
    #def create_basket_col(env):

    #    long_info1 = OpenRAVEBody.create_body_info(KinBody.Link.GeomType.Box, [.3,.15,.015], [0, 0.75, 1])
    #    long_info2 = OpenRAVEBody.create_body_info(KinBody.Link.GeomType.Box, [.3,.15,.015], [0, 0.75, 1])
    #    short_info1 = OpenRAVEBody.create_body_info(KinBody.Link.GeomType.Box, [.015,.15,.2], [0, 0.75, 1])
    #    short_info2 = OpenRAVEBody.create_body_info(KinBody.Link.GeomType.Box, [.015,.15,.2], [0, 0.75, 1])
    #    bottom_info = OpenRAVEBody.create_body_info(KinBody.Link.GeomType.Box, [.3,.015,.2], [0, 0.75, 1])

    #    long_info1._t = OpenRAVEBody.transform_from_obj_pose([0,-0.118,0.208],[0,0,0.055])
    #    long_info2._t = OpenRAVEBody.transform_from_obj_pose([0,-0.118,-0.208],[0,0,-0.055])
    #    short_info1._t = OpenRAVEBody.transform_from_obj_pose([0.309,-0.118,0],[-0.055,0,0])
    #    short_info2._t = OpenRAVEBody.transform_from_obj_pose([-0.309,-0.118,0],[0.055,0,0])
    #    bottom_info._t = OpenRAVEBody.transform_from_obj_pose([0,-0.25,0],[0,0,0])
    #    basket = RaveCreateRobot(env, '')
    #    basket.InitFromGeometries([long_info1, long_info2, short_info1, short_info2, bottom_info])
    #    return basket

    #@staticmethod
    #def create_table(env, geom):
    #    thickness = geom.thickness
    #    leg_height = geom.leg_height
    #    back = geom.back
    #    dim1, dim2 = geom.table_dim
    #    legdim1, legdim2 = geom.leg_dim

    #    table_color = [0.5, 0.2, 0.1]
    #    component_type = KinBody.Link.GeomType.Box
    #    tabletop = OpenRAVEBody.create_body_info(component_type, [dim1/2, dim2/2, thickness/2], table_color)

    #    leg1 = OpenRAVEBody.create_body_info(component_type, [legdim1/2, legdim2/2, leg_height/2], table_color)
    #    leg1._t[0, 3] = dim1/2 - legdim1/2
    #    leg1._t[1, 3] = dim2/2 - legdim2/2
    #    leg1._t[2, 3] = -leg_height/2 - thickness/2

    #    leg2 = OpenRAVEBody.create_body_info(component_type, [legdim1/2, legdim2/2, leg_height/2], table_color)
    #    leg2._t[0, 3] = dim1/2 - legdim1/2
    #    leg2._t[1, 3] = -dim2/2 + legdim2/2
    #    leg2._t[2, 3] = -leg_height/2 - thickness/2

    #    leg3 = OpenRAVEBody.create_body_info(component_type, [legdim1/2, legdim2/2, leg_height/2], table_color)
    #    leg3._t[0, 3] = -dim1/2 + legdim1/2
    #    leg3._t[1, 3] = dim2/2 - legdim2/2
    #    leg3._t[2, 3] = -leg_height/2 - thickness/2

    #    leg4 = OpenRAVEBody.create_body_info(component_type, [legdim1/2, legdim2/2, leg_height/2], table_color)
    #    leg4._t[0, 3] = -dim1/2 + legdim1/2
    #    leg4._t[1, 3] = -dim2/2 + legdim2/2
    #    leg4._t[2, 3] = -leg_height/2 - thickness/2

    #    if back:
    #        back_plate = OpenRAVEBody.create_body_info(component_type, [legdim1/10, dim2/2, leg_height-thickness/2], table_color)
    #        back_plate._t[0, 3] = dim1/2 - legdim1/10
    #        back_plate._t[1, 3] = 0
    #        back_plate._t[2, 3] = -leg_height/2 - thickness/4

    #    table = RaveCreateRobot(env, '')
    #    if not back:
    #        table.InitFromGeometries([tabletop, leg1, leg2, leg3, leg4])
    #    else:
    #        table.InitFromGeometries([tabletop, leg1, leg2, leg3, leg4, back_plate])
    #    return table

    @staticmethod
    def base_pose_2D_to_mat(pose):
        # x, y = pose
        assert len(pose) == 2
        x = pose[0]
        y = pose[1]
        pos = [x, y, 0]
        rot = 0
        matrix = T.pose2mat((pos, [1, 0, 0, 0]))
        return matrix

    @staticmethod
    def base_pose_3D_to_mat(pose):
        # x, y, z = pose
        assert len(pose) == 3
        x = pose[0]
        y = pose[1]
        z = pose[2]
        pos = [x, y, z]
        rot = 0
        matrix = T.pose2mat((pos, [0, 0, 0, 1]))
        return matrix

    @staticmethod
    def mat_to_base_pose_2D(mat):
        return T.mat2pose(mat)[0][:2]

    @staticmethod
    def base_pose_to_mat(pose):
        # x, y, rot = pose
        assert len(pose) == 3
        x = pose[0]
        y = pose[1]
        rot = pose[2]
        pos = [x, y, 0]
        quat = T.euler_to_quaternion([0, 0, rot], order='xyzw')
        matrix = T.pose2mat((pos, quat))
        return matrix

    # @staticmethod
    # def angle_pose_to_mat(pose):
    #     assert len(pose) == 1
    #     if USE_OPENRAVE:
    #         q = quatFromAxisAngle((0, 0, pose)).tolist()
    #         matrix = matrixFromPose(q + pos)
    #     else:
    #         quat = T.euler_to_quaternion([0, 0, pose], order='xyzw')
    #         matrix = T.pose2mat((pos, quat))

    #     return matrix

    @staticmethod
    def mat_to_base_pose(mat):
        pose = poseFromMatrix(mat)
        x = pose[4]
        y = pose[5]
        rot = axisAngleFromRotationMatrix(mat)[2]
        return np.array([x,y,rot])

    @staticmethod
    def obj_pose_from_transform(transform):
        trans = transform[:3,3]
        rot_matrix = transform[:3,:3]
        yaw, pitch, roll = OpenRAVEBody._ypr_from_rot_matrix(rot_matrix)
        return np.array((trans[0], trans[1], trans[2], yaw, pitch, roll))

    @staticmethod
    def transform_from_obj_pose(pose, rotation = np.array([0,0,0])):
        x, y, z = pose
        if len(rotation) == 4:
            rotation = T.quaternion_to_euler(rotation, order='xyzw')
            rotation = [rotation[2], rotation[1], rotation[0]]
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(pose, rotation)
        rot_mat = np.dot(Rz, np.dot(Ry, Rx))
        matrix = np.eye(4)
        matrix[:3,:3] = rot_mat
        matrix[:3,3] = [x,y,z]
        return matrix

    @staticmethod
    def _axis_rot_matrices(pose, rotation):
        x, y, z = pose
        alpha, beta, gamma = rotation
        Rz_2d = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
        Ry_2d = np.array([[cos(beta), sin(beta)], [-sin(beta), cos(beta)]])
        Rx_2d = np.array([[cos(gamma), -sin(gamma)], [sin(gamma), cos(gamma)]])
        I = np.eye(3)
        Rz = I.copy()
        Rz[:2,:2] = Rz_2d
        Ry = I.copy()
        Ry[[[0],[2]],[0,2]] = Ry_2d
        Rx = I.copy()
        Rx[1:3,1:3] = Rx_2d
        return Rz, Ry, Rx

    @staticmethod
    def _ypr_from_rot_matrix(r):
        # alpha
        yaw = atan2(r[1,0], r[0,0])
        # beta
        pitch = atan2(-r[2,0],np.sqrt(r[2,1]**2+r[2,2]**2))
        # gamma
        roll = atan2(r[2,1], r[2,2])
        return (yaw, pitch, roll)

    @staticmethod
    def quat_from_v1_to_v2(v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        xyz = np.cross(v1, v2)
        len1 = np.sum(v1**2)
        len2 = np.sum(v2**2)
        w = np.sqrt(len1 * len2) + np.dot(v1, v2)
        quat = np.concatenate([xyz, [w]])
        if np.all(np.abs(quat) < 1e-5):
            v1 = v1 / np.linalg.norm(v1)
            mid_axis = [1, 0, 0] if np.abs(np.dot([1,0,0], v1)) < 1-1e-3 else [0, 1, 0]
            quat1 = OpenRAVEBody.quat_from_v1_to_v2(v1, mid_axis)
            quat2 = OpenRAVEBody.quat_from_v1_to_v2(mid_axis, v2)
            mat1 = T.quat2mat(quat1)
            mat2 = T.quat2mat(quat2)
            quat = T.mat2quat(mat1.dot(mat2))
        quat /= np.linalg.norm(quat)
        return quat

    #@staticmethod
    #def get_ik_transform(pos, rot, right_arm = True):
    #    trans = OpenRAVEBody.transform_from_obj_pose(pos, rot)
    #    # Openravepy flip the rotation axis by 90 degree, thus we need to change it back
    #    if right_arm:
    #        quat = T.euler_to_quaternion([0, np.pi/2, 0], order='xyzw')
    #    else:
    #        quat = T.euler_to_quaternion([0, -np.pi/2, 0], order='xyzw')
    #    rot_mat = T.pose2mat([[0, 0, 0], quat])
    #    trans_mat = trans[:3, :3].dot(rot_mat[:3, :3])
    #    trans[:3, :3] = trans_mat
    #    return trans

    def get_link_pose(self, link_id, euler=True):
        info = P.getLinkState(self.body_id, link_id)
        pos, orn = info[0], info[1]
        if euler:
            orn = T.quaternion_to_euler(orn, order='xyzw')
        return pos, orn

    def current_pose(self, euler=True):
        pos, orn = P.getBasePositionAndOrientation(self.body_id)
        if euler:
            orn = T.quaternion_to_euler(orn, order='xyzw')
        return pos, orn

    def set_from_param(self, param, t):
        if param.is_symbol(): t = 0
        pos = param.pose[:,t] if not param.is_symbol() else param.value[:,0]
        if 'Robot' in param.get_type(True) or 'RobotPose' in param.get_type(True):
            dof_map = {}
            geom = param.openrave_body._geom
            for arm in geom.arms:
                dof_map[arm] = getattr(param, arm)[:,t]
            for gripper in geom.grippers:
                dof_map[gripper] = getattr(param, gripper)[:,t]
            self.set_dof(dof_map)
            self.set_pose(pos)
        else:
            self.set_pose(pos, param.rotation[:,t])

    def get_ik_from_pose(self, pos, rot, manip_name, use6d=True, multiple=0, maxIter=1024, bnds=None):
        quat = rot if (rot is None or len(rot) == 4) else T.euler_to_quaternion(rot, order='xyzw')
        pos = np.array(pos).tolist()
        quat = np.array(quat).tolist()
        if bnds is None:
            lb, ub = self._geom.get_arm_bnds()
        else:
            true_lb, true_ub = self._geom.get_arm_bnds()
            lb, ub = bnds
            if len(lb) < len(true_lb):
                lb = np.r_[lb, -10*np.ones(len(true_lb) - len(lb))]

            if len(ub) < len(true_ub):
                ub = np.r_[ub, 10*np.ones(len(true_ub) - len(ub))]

            lb = np.maximum(lb, true_lb).tolist()
            ub = np.minimum(ub, true_ub).tolist()

        ranges = (np.array(ub) - np.array(lb)).tolist()
        jnt_ids = sorted(self._geom.get_free_inds())
        jnts = P.getJointStates(self.body_id, jnt_ids)
        rest_poses = [j[0] for j in jnts]
        cur_jnts = rest_poses
        manip_id = self._geom.get_ee_link(manip_name)
        damp = (0.1 * np.ones(len(jnt_ids))).tolist()
        joint_pos = P.calculateInverseKinematics(self.body_id,
                                                 manip_id, 
                                                 pos,
                                                 quat,
                                                 lowerLimits=lb,
                                                 upperLimits=ub,
                                                 jointRanges=ranges,
                                                 restPoses=rest_poses,
                                                 jointDamping=damp,
                                                 maxNumIterations=maxIter)
        inds = list(self._geom.get_free_inds(manip_name))
        joint_pos = np.array(joint_pos)[inds].tolist()
        lb, ub = self._geom.get_joint_limits(manip_name)
        joint_pos = np.maximum(np.minimum(joint_pos, ub), lb)
        if not multiple: return joint_pos
        poses = [joint_pos]
        for _ in range(multiple):
            rest_poses = (np.array(cur_jnts) + 5 * (np.random.uniform(size=len(lb)) - 0.5) * ranges).tolist()
            joint_pos = P.calculateInverseKinematics(self.body_id,
						     manip_id,
						     pos,
						     quat,
						     lb,
						     ub,
						     ranges,
						     rest_poses,
						     maxNumIterations=maxIter)
            joint_pos = np.array(joint_pos)[inds].tolist()
            poses.append(joint_pos)
        return poses

    def fwd_kinematics(self, manip_name, dof_map=None, mat_result=False):
        if dof_map is not None:
            self.set_dof(dof_map)

        ee_link = self._geom.get_ee_link(manip_name)
        link_state = P.getLinkState(self.body_id, ee_link)
        pos = link_state[0]
        quat = link_state[1]
        if mat_result:
            return OpenRAVEBody.transform_from_obj_pose(pos, quat)
        return {'pos': pos, 'quat': quat}

    def param_fwd_kinematics(self, param, manip_names, t, mat_result=False):
        if not isinstance(self._geom, Robot): return

        attrs = list(param._attr_types.keys())
        attr_vals = {attr: getattr(param, attr)[:, t] for attr in attrs if attr in self._geom.dof_map}
        param.openrave_body.set_dof(attr_vals)

        result = {}
        for manip_name in manip_names:
            result[manip_name] = self.fwd_kinematics(manip_name, mat_result=mat_result)

        return result
