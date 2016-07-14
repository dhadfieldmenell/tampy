import numpy as np
from errors_exceptions import OpenRAVEException
from openravepy import quatFromAxisAngle, matrixFromPose, poseFromMatrix, \
axisAngleFromRotationMatrix, KinBody, GeometryType, RaveCreateRobot, \
RaveCreateKinBody, TriMesh, Environment
from core.util_classes.pr2 import PR2
from core.util_classes.can import Can, BlueCan, RedCan
from core.util_classes.circle import Circle, BlueCircle, RedCircle, GreenCircle
from core.util_classes.obstacle import Obstacle
from core.util_classes.table import Table

class OpenRAVEBody(object):
    def __init__(self, env, name, geom):
        assert env is not None
        self.name = name
        self._env = env
        self._geom = geom
        if isinstance(geom, Circle):
            self._add_circle(geom)
        elif isinstance(geom, Can):
            self._add_circle(geom)
        elif isinstance(geom, Obstacle):
            self._add_obstacle()
        elif isinstance(geom, PR2):
            self._add_robot(geom)
        elif isinstance(geom, Table):
            self._add_table(geom)
        else:
            raise OpenRAVEException("Geometry not supported for %s for OpenRAVEBody"%geom)


    def _add_circle(self, geom):
        color = None
        if hasattr(geom, "color") and geom.color == 'blue':
            color = [0, 0, 1]
        elif hasattr(geom, "color") and geom.color == 'green':
            color = [0, 1, 0]
        elif hasattr(geom, "color") and geom.color == 'red':
            color = [1, 0, 0]
        else:
            color = [1,0,0]
        self.env_body = OpenRAVEBody.create_cylinder(self._env, self.name, np.eye(4),
                [geom.radius, 2], color)
        self._env.AddKinBody(self.env_body)

    def _add_obstacle(self):
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

    def _add_robot(self, geom):
        self.env_body = self._env.ReadRobotXMLFile(geom.shape)
        self.env_body.SetName(self.name)
        self._env.Add(self.env_body)

    def _add_table(self, geom):
        self.env_body = OpenRAVEBody.create_table(self._env, geom)
        self.env_body.SetName(self.name)
        self._env.Add(self.env_body)

    def set_pose(self, base_pose):
        trans = None
        if isinstance(self._geom, Circle):
            trans = OpenRAVEBody.base_pose_2D_to_mat(base_pose)
        elif isinstance(self._geom, Obstacle):
            trans = OpenRAVEBody.base_pose_2D_to_mat(base_pose)
        elif isinstance(self._geom, PR2):
            trans = OpenRAVEBody.base_pose_to_mat(base_pose)
        elif isinstance(self._geom, Table):
            trans = OpenRAVEBody.base_pose_3D_to_mat(base_pose)
        self.env_body.SetTransform(trans)

    def set_dof(self, back_height, l_arm_pose, r_arm_pose):
        """
            This function assumed to be called when the self.env_body is a robot and its geom is type PR2
            It sets the DOF values for important joint of PR2

            back_height: back_height attribute of type Value, which specified the back height of PR2
            l_arm_pose: l_arm_pose attribute of type Vector8d, which specified the left arm pose of PR2
            r_arm_pose: r_arm_pose attribute of type Vector8d, which specified the right arm pose of PR2
        """
        dof_val = self.env_body.GetActiveDOFValues()
        back_height_index = self.env_body.GetJoint('torso_lift_joint').GetDOFIndex()
        l_shoulder_index = self.env_body.GetJoint('l_shoulder_pan_joint').GetDOFIndex()
        r_shoulder_index = self.env_body.GetJoint('r_shoulder_pan_joint').GetDOFIndex()

        dof_val[back_height_index] = back_height[0]
        dof_val[l_shoulder_index: l_shoulder_index + 8] = l_arm_pose.reshape((1, 8)).tolist()[0]
        dof_val[r_shoulder_index: r_shoulder_index + 8] = r_arm_pose.reshape((1, 8)).tolist()[0]
        self.env_body.SetActiveDOFValues(dof_val)

    @staticmethod
    def create_cylinder(env, body_name, t, dims, color=[0, 1, 1]):
        infocylinder = OpenRAVEBody.create_body_info(GeometryType.Cylinder, dims, color)
        if type(env) != Environment:
            import ipdb; ipdb.set_trace()
        cylinder = RaveCreateKinBody(env, '')
        cylinder.InitFromGeometries([infocylinder])
        cylinder.SetName(body_name)
        cylinder.SetTransform(t)
        return cylinder

    @staticmethod
    def create_box(env, name, transform, dims, color=[0,0,1]):
        infobox = OpenRAVEBody.create_box_info(dims, color, 0, True)
        box = RaveCreateKinBody(env,'')
        box.InitFromGeometries([infobox])
        box.SetName(name)
        box.SetTransform(transform)
        return box

    @staticmethod
    def create_body_info(body_type, dims, color, transparency = 0.0, visible = True):
        infobox = KinBody.Link.GeometryInfo()
        infobox._type = body_type
        infobox._vGeomData = dims
        infobox._bVisible = True
        infobox._fTransparency = 0
        infobox._vDiffuseColor = color
        return infobox

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
        # x, y = pose
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
    def mat_to_base_pose(mat):
        pose = poseFromMatrix(mat)
        x = pose[4]
        y = pose[5]
        rot = axisAngleFromRotationMatrix(mat)[2]
        return np.array([x,y,rot])
