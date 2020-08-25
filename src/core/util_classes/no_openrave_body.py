import numpy as np
from math import cos, sin, atan2
from errors_exceptions import OpenRAVEException
from openravepy import quatFromAxisAngle, matrixFromPose, poseFromMatrix, \
axisAngleFromRotationMatrix, KinBody, GeometryType, RaveCreateRobot, \
RaveCreateKinBody, TriMesh, Environment, DOFAffine, IkParameterization, IkParameterizationType, \
IkFilterOptions, matrixFromAxisAngle, quatFromRotationMatrix
from core.util_classes.robots import Robot, PR2, Baxter, Washer

from core.util_classes.items import Item, Box, Can, BlueCan, RedCan, Circle, BlueCircle, RedCircle, GreenCircle, Obstacle, Wall, Table, Basket

WALL_THICKNESS = 1

class OpenRAVEBody(object):
    def __init__(self, env, name, geom):
        assert env is not None
        self.name = name
        self._env = env
        self._geom = geom

        if env.GetKinBody(name) == None and env.GetRobot(name) == None:
            if isinstance(geom, Robot):
                self._add_robot(geom)
            elif  isinstance(geom, Item):
                self._add_item(geom)
            else:
                raise OpenRAVEException("Geometry not supported for %s for OpenRAVEBody"%geom)
        elif env.GetKinBody(name) != None:
            self.env_body = env.GetKinBody(name)
        else:
            self.env_body = env.GetRobot(name)
        self.set_transparency(0.5)

    def delete(self):
        self._env.Remove(self.env_body)

    def set_transparency(self, transparency):
        for link in self.env_body.GetLinks():
            for geom in link.GetGeometries():
                geom.SetTransparency(transparency)

    def _add_robot(self, geom):
        self.env_body = self._env.ReadRobotXMLFile(geom.shape)
        self.env_body.SetName(self.name)
        self._env.Add(self.env_body)
        geom.setup(self.env_body)

    def _add_item(self, geom):
        try:
            fun_name = "self._add_{}".format(geom._type)
            eval(fun_name)(geom)
        except:
            self._add_obj(geom)

    def _add_circle(self, geom):
        color = [1,0,0]
        if hasattr(geom, "color") and geom.color == 'blue':
            color = [0, 0, 1]
        elif hasattr(geom, "color") and geom.color == 'green':
            color = [0, 1, 0]
        elif hasattr(geom, "color") and geom.color == 'red':
            color = [1, 0, 0]

        self.env_body = OpenRAVEBody.create_cylinder(self._env, self.name, np.eye(4),
                [geom.radius, 2], color)
        self._env.AddKinBody(self.env_body)

    def _add_can(self, geom):
        color = [1,0,0]
        if hasattr(geom, "color") and geom.color == 'blue':
            color = [0, 0, 1]
        elif hasattr(geom, "color") and geom.color == 'green':
            color = [0, 1, 0]
        elif hasattr(geom, "color") and geom.color == 'red':
            color = [1, 0, 0]

        self.env_body = OpenRAVEBody.create_cylinder(self._env, self.name, np.eye(4),
                [geom.radius, geom.height], color)
        self._env.AddKinBody(self.env_body)

    def _add_obstacle(self, geom):
        obstacles = np.matrix('-0.576036866359447, 0.918128654970760, 1;\
                        -0.806451612903226,-1.07017543859649, 1;\
                        1.01843317972350,-0.988304093567252, 1;\
                        0.640552995391705,0.906432748538011, 1;\
                        -0.576036866359447, 0.918128654970760, -1;\
                        -0.806451612903226,-1.07017543859649, -1;\
                        1.01843317972350,-0.988304093567252, -1;\
                        0.640552995391705,0.906432748538011, -1')

        body = RaveCreateKinBody(self._env, '')
        vertices = np.array(obstacles)
        indices = np.array([[0, 1, 2], [2, 3, 0], [4, 5, 6], [6, 7, 4], [0, 4, 5],
                            [0, 1, 5], [1, 2, 5], [5, 6, 2], [2, 3, 6], [6, 7, 3],
                            [0, 3, 7], [0, 4, 7]])
        body.InitFromTrimesh(trimesh=TriMesh(vertices, indices), draw=True)
        body.SetName(self.name)
        for link in body.GetLinks():
            for geom in link.GetGeometries():
                geom.SetDiffuseColor((.9, .9, .9))
        self.env_body = body
        self._env.AddKinBody(body)

    def _add_box(self, geom):
        infobox = OpenRAVEBody.create_body_info(KinBody.Link.GeomType.Box, geom.dim, [0.5, 0.2, 0.1])
        self.env_body = RaveCreateKinBody(self._env,'')
        self.env_body.InitFromGeometries([infobox])
        self.env_body.SetName(self.name)
        self._env.Add(self.env_body)

    def _add_sphere(self, geom):
        infobox = OpenRAVEBody.create_body_info(KinBody.Link.GeomType.Sphere, [geom.radius], [0, 0, 1])
        self.env_body = RaveCreateKinBody(self._env,'')
        self.env_body.InitFromGeometries([infobox])
        self.env_body.SetName(self.name)
        self._env.Add(self.env_body)

    def _add_wall(self, geom):
        self.env_body = OpenRAVEBody.create_wall(self._env, geom.wall_type)
        self.env_body.SetName(self.name)
        self._env.Add(self.env_body)

    def _add_obj(self, geom):
        self.env_body = self._env.ReadKinBodyXMLFile(geom.shape)
        self.env_body.SetName(self.name)
        self._env.Add(self.env_body)

    def _add_table(self, geom):
        self.env_body = OpenRAVEBody.create_table(self._env, geom)
        self.env_body.SetName(self.name)
        self._env.Add(self.env_body)

    def _add_basket(self, geom):
        self.env_body = self._env.ReadKinBodyXMLFile(geom.shape)
        self.env_body.SetName(self.name)
        self._env.Add(self.env_body)

    def set_pose(self, base_pose, rotation = [0, 0, 0]):
        trans = None
        if np.any(np.isnan(base_pose)) or np.any(np.isnan(rotation)):
            return
        if isinstance(self._geom, Robot) and not isinstance(self._geom, Washer):
            trans = OpenRAVEBody.base_pose_to_mat(base_pose)
        elif len(base_pose) == 2:
            trans = OpenRAVEBody.base_pose_2D_to_mat(base_pose)
        else:
            trans = OpenRAVEBody.transform_from_obj_pose(base_pose, rotation)
        self.env_body.SetTransform(trans)

    def set_dof(self, dof_value_map):
        """
            dof_value_map: A dict that maps robot attribute name to a list of corresponding values
        """
        # make sure only sets dof for robot
        # assert isinstance(self._geom, Robot)
        if not isinstance(self._geom, Robot): return

        # Get current dof value for each joint
        dof_val = self.env_body.GetActiveDOFValues()

        for k, v in dof_value_map.items():
            if k not in self._geom.dof_map or np.any(np.isnan(v)): continue
            inds = self._geom.dof_map[k]
            try:
                dof_val[inds] = v
            except IndexError:
                print(('\n\n\nBad index in set dof:', inds, k, v, self._geom, '\n\n\n'))
        # Set new DOF value to the robot
        self.env_body.SetActiveDOFValues(dof_val)

    def _set_active_dof_inds(self, inds = None):
        """
        Set active dof index to the one we are interested
        This function is implemented to simplify jacobian calculation in the CollisionPredicate
        inds: Optional list of index specifying dof index we are interested in
        """
        robot = self.env_body
        if inds == None:
            dof_inds = np.ndarray(0, dtype=np.int)
            if robot.GetJoint("torso_lift_joint") != None:
                dof_inds = np.r_[dof_inds, robot.GetJoint("torso_lift_joint").GetDOFIndex()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("leftarm").GetArmIndices()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("leftarm").GetGripperIndices()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("rightarm").GetArmIndices()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("rightarm").GetGripperIndices()]
            robot.SetActiveDOFs(
                                dof_inds,
                                DOFAffine.X + DOFAffine.Y + DOFAffine.RotationAxis,
                                [0, 0, 1])
        else:
            robot.SetActiveDOFs(inds)

    @staticmethod
    def create_cylinder(env, body_name, t, dims, color=[0, 1, 1]):
        infocylinder = OpenRAVEBody.create_body_info(GeometryType.Cylinder, dims, color)
        if type(env) != Environment:
            # import ipdb; ipdb.set_trace()
            print("Environment object is not valid")
        cylinder = RaveCreateKinBody(env, '')
        cylinder.InitFromGeometries([infocylinder])
        cylinder.SetName(body_name)
        cylinder.SetTransform(t)
        return cylinder

    @staticmethod
    def create_box(env, name, transform, dims, color=[0,0,1]):
        infobox = OpenRAVEBody.create_body_info(KinBody.Link.GeomType.Box, dims, color, 0, True)
        box = RaveCreateKinBody(env,'')
        box.InitFromGeometries([infobox])
        box.SetName(name)
        box.SetTransform(transform)
        return box

    @staticmethod
    def create_sphere(env, name, transform, dims, color=[0,0,1]):
        infobox = OpenRAVEBody.create_body_info(GeometryType.Sphere, dims, color)
        sphere = RaveCreateKinBody(env,'')
        sphere.InitFromGeometries([infobox])
        sphere.SetName(name)
        sphere.SetTransform(transform)
        return sphere

    @staticmethod
    def create_body_info(body_type, dims, color, transparency = 0.8, visible = True):
        infobox = KinBody.Link.GeometryInfo()
        infobox._type = body_type
        infobox._vGeomData = dims
        infobox._bVisible = True
        infobox._fTransparency = transparency
        infobox._vDiffuseColor = color
        return infobox

    @staticmethod
    def create_wall(env, wall_type):
        component_type = KinBody.Link.GeomType.Box
        wall_color = [0.5, 0.2, 0.1]
        box_infos = []
        if wall_type == 'closet':
            wall_endpoints = [[-6.0,-8.0],[-6.0,4.0],[1.9,4.0],[1.9,8.0],[5.0,8.0],[5.0,4.0],[13.0,4.0],[13.0,-8.0],[-6.0,-8.0]]
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
            box_info = OpenRAVEBody.create_body_info(component_type, dims, wall_color)
            box_info._t = transform
            box_infos.append(box_info)
        wall = RaveCreateKinBody(env, '')
        wall.InitFromGeometries(box_infos)
        return wall


    @staticmethod
    def get_wall_dims(wall_type='closet'):
        wall_endpoints = [[-6.0,-8.0],[-6.0,4.0],[1.9,4.0],[1.9,8.0],[5.0,8.0],[5.0,4.0],[13.0,4.0],[13.0,-8.0],[-6.0,-8.0]]
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


    @staticmethod
    def create_basket_col(env):

        long_info1 = OpenRAVEBody.create_body_info(KinBody.Link.GeomType.Box, [.3,.15,.015], [0, 0.75, 1])
        long_info2 = OpenRAVEBody.create_body_info(KinBody.Link.GeomType.Box, [.3,.15,.015], [0, 0.75, 1])
        short_info1 = OpenRAVEBody.create_body_info(KinBody.Link.GeomType.Box, [.015,.15,.2], [0, 0.75, 1])
        short_info2 = OpenRAVEBody.create_body_info(KinBody.Link.GeomType.Box, [.015,.15,.2], [0, 0.75, 1])
        bottom_info = OpenRAVEBody.create_body_info(KinBody.Link.GeomType.Box, [.3,.015,.2], [0, 0.75, 1])

        long_info1._t = OpenRAVEBody.transform_from_obj_pose([0,-0.118,0.208],[0,0,0.055])
        long_info2._t = OpenRAVEBody.transform_from_obj_pose([0,-0.118,-0.208],[0,0,-0.055])
        short_info1._t = OpenRAVEBody.transform_from_obj_pose([0.309,-0.118,0],[-0.055,0,0])
        short_info2._t = OpenRAVEBody.transform_from_obj_pose([-0.309,-0.118,0],[0.055,0,0])
        bottom_info._t = OpenRAVEBody.transform_from_obj_pose([0,-0.25,0],[0,0,0])
        basket = RaveCreateRobot(env, '')
        basket.InitFromGeometries([long_info1, long_info2, short_info1, short_info2, bottom_info])
        return basket

    @staticmethod
    def create_table(env, geom):
        thickness = geom.thickness
        leg_height = geom.leg_height
        back = geom.back
        dim1, dim2 = geom.table_dim
        legdim1, legdim2 = geom.leg_dim

        table_color = [0.5, 0.2, 0.1]
        component_type = KinBody.Link.GeomType.Box
        tabletop = OpenRAVEBody.create_body_info(component_type, [dim1/2, dim2/2, thickness/2], table_color)

        leg1 = OpenRAVEBody.create_body_info(component_type, [legdim1/2, legdim2/2, leg_height/2], table_color)
        leg1._t[0, 3] = dim1/2 - legdim1/2
        leg1._t[1, 3] = dim2/2 - legdim2/2
        leg1._t[2, 3] = -leg_height/2 - thickness/2

        leg2 = OpenRAVEBody.create_body_info(component_type, [legdim1/2, legdim2/2, leg_height/2], table_color)
        leg2._t[0, 3] = dim1/2 - legdim1/2
        leg2._t[1, 3] = -dim2/2 + legdim2/2
        leg2._t[2, 3] = -leg_height/2 - thickness/2

        leg3 = OpenRAVEBody.create_body_info(component_type, [legdim1/2, legdim2/2, leg_height/2], table_color)
        leg3._t[0, 3] = -dim1/2 + legdim1/2
        leg3._t[1, 3] = dim2/2 - legdim2/2
        leg3._t[2, 3] = -leg_height/2 - thickness/2

        leg4 = OpenRAVEBody.create_body_info(component_type, [legdim1/2, legdim2/2, leg_height/2], table_color)
        leg4._t[0, 3] = -dim1/2 + legdim1/2
        leg4._t[1, 3] = -dim2/2 + legdim2/2
        leg4._t[2, 3] = -leg_height/2 - thickness/2

        if back:
            back_plate = OpenRAVEBody.create_body_info(component_type, [legdim1/10, dim2/2, leg_height-thickness/2], table_color)
            back_plate._t[0, 3] = dim1/2 - legdim1/10
            back_plate._t[1, 3] = 0
            back_plate._t[2, 3] = -leg_height/2 - thickness/4

        table = RaveCreateRobot(env, '')
        if not back:
            table.InitFromGeometries([tabletop, leg1, leg2, leg3, leg4])
        else:
            table.InitFromGeometries([tabletop, leg1, leg2, leg3, leg4, back_plate])
        return table

    @staticmethod
    def base_pose_2D_to_mat(pose):
        # x, y = pose
        assert len(pose) == 2
        x = pose[0]
        y = pose[1]
        rot = 0
        q = quatFromAxisAngle((0, 0, rot)).tolist()
        pos = [x, y, 0]
        # pos = np.vstack((x,y,np.zeros(1)))
        matrix = matrixFromPose(q + pos)
        return matrix

    @staticmethod
    def base_pose_3D_to_mat(pose):
        # x, y, z = pose
        assert len(pose) == 3
        x = pose[0]
        y = pose[1]
        z = pose[2]
        rot = 0
        q = quatFromAxisAngle((0, 0, rot)).tolist()
        pos = [x, y, z]
        # pos = np.vstack((x,y,np.zeros(1)))
        matrix = matrixFromPose(q + pos)
        return matrix

    @staticmethod
    def mat_to_base_pose_2D(mat):
        pose = poseFromMatrix(mat)
        x = pose[4]
        y = pose[5]
        return np.array([x,y])

    @staticmethod
    def base_pose_to_mat(pose):
        # x, y, rot = pose
        assert len(pose) == 3
        x = pose[0]
        y = pose[1]
        rot = pose[2]
        q = quatFromAxisAngle((0, 0, rot)).tolist()
        pos = [x, y, 0]
        # pos = np.vstack((x,y,np.zeros(1)))
        matrix = matrixFromPose(q + pos)
        return matrix

    @staticmethod
    def angle_pose_to_mat(pose):
        assert len(pose) == 1
        q = quatFromAxisAngle((0, 0, pose)).tolist()
        matrix = matrixFromPose(q + pos)
        return matrix

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
        # ipdb.set_trace()
        return np.array((trans[0], trans[1], trans[2], yaw, pitch, roll))

    @staticmethod
    def transform_from_obj_pose(pose, rotation = np.array([0,0,0])):
        x, y, z = pose
        alpha, beta, gamma = rotation
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
        # ipdb.set_trace()
        return Rz, Ry, Rx

    @staticmethod
    def _ypr_from_rot_matrix(r):
        # alpha
        yaw = atan2(r[1,0], r[0,0])
        # beta
        pitch = atan2(-r[2,0],np.sqrt(r[2,1]**2+r[2,2]**2))
        # gamma
        roll = atan2(r[2,1], r[2,2])
        # ipdb.set_trace()
        return (yaw, pitch, roll)

    @staticmethod
    def get_ik_transform(pos, rot, right_arm = True):
        trans = OpenRAVEBody.transform_from_obj_pose(pos, rot)
        # Openravepy flip the rotation axis by 90 degree, thus we need to change it back
        if right_arm:
            rot_mat = matrixFromAxisAngle([0, np.pi/2, 0])
        else:
            rot_mat = matrixFromAxisAngle([0, -np.pi/2, 0])
        trans_mat = trans[:3, :3].dot(rot_mat[:3, :3])
        trans[:3, :3] = trans_mat
        return trans

    def get_ik_arm_pose(self, pos, rot):
        # assert isinstance(self._geom, PR2)
        solutions = self.get_ik_from_pose(pos, rot, 'rightarm_torso')
        return solutions

    def get_ik_from_pose(self, pos, rot, manip_name, use6d=True):
        trans = OpenRAVEBody.get_ik_transform(pos, rot)
        solutions = self.get_ik_solutions(manip_name, trans, use6d)
        return solutions

    def get_ik_solutions(self, manip_name, trans, use6d=True):
        manip = self.env_body.GetManipulator(manip_name)
        if use6d:
            iktype = IkParameterizationType.Transform6D
        else:
            iktype = IkParameterizationType.Translation3D
        solutions = manip.FindIKSolutions(IkParameterization(trans, iktype),IkFilterOptions.CheckEnvCollisions)
        return solutions

    def get_close_ik_solution(self, manip_name, trans, dof_map=None):
        if dof_map is not None:
            self.set_dof(dof_map)

        manip = self.env_body.GetManipulator(manip_name)
        iktype = IkParameterizationType.Transform6D
        ik_param = IkParameterization(trans, iktype)
        solution = manip.FindIKSolution(ik_param, IkFilterOptions.IgnoreSelfCollisions)
        return solution

    def fwd_kinematics(self, manip_name, dof_map=None, mat_result=False):
        if dof_map is not None:
            self.set_dof(dof_map)

        trans = self.env_body.GetLink(manip_name).GetTransform()
        if mat_result:
            return trans

        pos = trans[:3, 3]
        quat = quatFromRotationMatrix(trans[:3, :3])
        return {'pos': pos, 'quat': quat}

    def param_fwd_kinematics(self, param, manip_names, t, mat_result=False):
        if not isinstance(self._geom, Robot): return

        attrs = list(param._attr_types.keys())
        dof_val = self.env_body.GetActiveDOFValues()
        for attr in attrs:
            if attr not in self._geom.dof_map: continue
            val = getattr(param, attr)[:, t]
            if np.any(np.isnan(val)): continue
            inds = self._geom.dof_map[attr]
            dof_val[inds] = val
        self.env_body.SetActiveDOFValues(dof_val)

        result = {}
        for manip_name in manip_names:
            result[manip_name] = self.fwd_kinematics(manip_name, mat_result=mat_result)

        return result
