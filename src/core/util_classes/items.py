import pybullet as p
import core.util_classes.common_constants as const


class Item(object):
    """
    Base class of every Objects in environment
    """
    def __init__(self):
        self._type = "item"
        self._base_type = "item"
        self.col_links = set([-1])
        self.grasp_point = [0., 0., 0.]

    def get_types(self):
        return [self._type, self._base_type]


class XMLItem(Item):
    def __init__(self, shape):
        super(XMLItem, self).__init__()
        self.initialized = False
        self._type = "xml_item"
        self._base_type = "xml_item"
        self.shape = shape
        self.grasp_point = [0., 0., 0.]
        self.dof_map = {}

    def is_initialized(self):
        return self.initialized

"""
Object defined for NAMO Domain
"""

class Circle(Item):
    """
    Defines geometry used in the NAMO domain.
    """
    def __init__(self, radius):
        super(Circle, self).__init__()
        self._type = "circle"
        self.radius = float(radius)

class RedCircle(Circle):
    def __init__(self, radius):
        super(RedCircle, self).__init__(radius)
        self.color = "red"

class BlueCircle(Circle):
    def __init__(self, radius):
        super(BlueCircle, self).__init__(radius)
        self.color = "blue"

class GreenCircle(Circle):
    def __init__(self, radius):
        super(GreenCircle, self).__init__(radius)
        self.color = "green"

class Door2d(Item):
    """
    Defines geometry used in the NAMO domain.
    """
    def __init__(self, radius, length):
        super(Door2d, self).__init__()
        self.dof_map = {'door_hinge': 0}
        self._type = "door"
        self.radius = float(radius)
        self.length = float(length)

class Wall(Item):

    """
    Defines the Wall class as geometry of Obstacle used in the Namo domain.
    """

    def __init__(self, wall_type):
        super(Wall, self).__init__()
        self._type = "wall"
        self.wall_type = wall_type

"""
Obejct defined for Robot[baxter/pr2] domain
"""

class Can(Item):
    """
    Defines geometry used in the CAN domain.
    """
    def __init__(self, radius, height):
        super(Can, self).__init__()
        self._type = "can"
        self.radius = float(radius)
        self.height = float(height)
        z = max(0, self.height - 0.03)
        self.grasp_point = [0., 0., z]

class BlueCan(Can):
    def __init__(self, radius, height):
        super(BlueCan, self).__init__(radius, height)
        self.color = "blue"

class RedCan(Can):
    def __init__(self, radius, height):
        super(RedCan, self).__init__(radius, height)
        self.color = "red"

class GreenCan(Can):
    def __init__(self, radius, height):
        super(GreenCan, self).__init__(radius, height)
        self.color = "green"

class Cloth(Can):
    def __init__(self):
        super(Cloth, self).__init__(0.02,0.04)
        self.color = "blue"
        self._type = "can"

class Edge(Can):
    def __init__(self, length):
        super(Cloth, self).__init__(0.01,length)
        self.color = "red"
        self._type = "can"

class Sphere(Item):
    def __init__(self, radius):
        super(Sphere, self).__init__()
        self.color = "blue"
        self._type = "sphere"
        self.radius = float(radius)

class Obstacle(Item):
    """
    Defines geometry used for testing move with obstructs in the NAMO domain.
    """
    def __init__(self):
        super(Obstacle, self).__init__()
        self._type = "obstacle"

class Table(Obstacle):
    """
        Object stores all the information to for a table model
    """

    def __init__(self, dim):
        super(Table, self).__init__()
        self._type = "table"
        if isinstance(dim, str):
            dim = list(eval(dim))
        self.table_dim = [dim[0], dim[1]]
        self.thickness = dim[2]
        self.leg_dim = [dim[3], dim[4]]
        self.leg_height = dim[5]
        self.back = dim[6]

class Box(Obstacle):
    """
        Object stores all the information to for a box model
    """

    def __init__(self, dim):
        super(Box, self).__init__()
        self._type = "box"
        if isinstance(dim, str):
            dim = list(eval(dim))
        self.dim = dim
        self.length = dim[0]
        self.width = dim[1]
        self.height = dim[2]
        z = max(0, self.height - 0.03)
        self.grasp_point = [0., 0., z]

class Basket(Item):
    """
        Object stores all the information to for a Basket model
    """

    def __init__(self):
        super(Basket, self).__init__()
        self._type = "basket"
        self.shape = "../models/baxter/basket.xml"
        self.up_right_rot = [0, 0, 1.57]

        # self.col_links = set(['long_1', 'long_2', 'short_1', 'short_2', 'bottom'])

class Door(XMLItem):
    def __init__(self, door_type):
        import baxter_gym

        self.handle_orn = [0., 0., 0.]
        self.in_orn = [0., 0., 0.]
        if door_type.lower() == 'desk_drawer':
            shape = baxter_gym.__path__[0] + '/robot_info/robodesk/desk_drawer.xml'
            #self.handle_pos = [0., -0.36, 0.01]
            self.handle_pos = const.DRAWER_HANDLE_POS
            self.handle_orn = const.DRAWER_HANDLE_ORN
            #self.in_pos = [0., -0.2, 0.05]
            self.in_pos = const.IN_DRAWER_POS
            self.in_orn = const.IN_DRAWER_ORN
            self.hinge_type = 'prismatic'
            self.close_val = 0.
            self.open_val = -0.16 # -0.18 #-0.48
            self.open_thresh = -0.14
            self.close_thresh = -0.1
            self.open_dir = [0., -1., 0.]
        elif door_type.lower() == 'desk_shelf':
            shape = baxter_gym.__path__[0] + '/robot_info/robodesk/desk_shelf.xml'
            self.hinge_type = 'prismatic'
            self.handle_pos = const.SHELF_HANDLE_POS 
            self.in_pos = const.IN_SHELF_POS
            self.handle_orn = const.SHELF_HANDLE_ORN
            self.in_orn = const.IN_SHELF_ORN
            self.close_val = 0.6
            self.open_val = 0.
            self.open_thresh = 0.3
            self.close_thresh = 0.5
            self.open_dir = [-1., 0., 0.]
        else:
            raise NotImplementedError()

        super(Door, self).__init__(shape)
        self._type = "door"

    def setup(self, robot=None):
        if self.initialized: return

        import pybullet as P
        self.initialized = True
        self.id = p.loadMJCF(self.shape)
        if type(self.id) is not int:
            for i in range(len(self.id)):
                if p.getNumJoints(self.id[i]) > 0:
                    self.id = self.id[i]
                    break

        njnt = p.getNumJoints(self.id)
        for jnt in range(njnt):
            info = p.getJointInfo(self.id, jnt)
            if info[2] != p.JOINT_FIXED:
                self.hinge = info[1]
                self.hinge_jnt = jnt
                self.dof_map['hinge'] = jnt
                if info[2] == p.JOINT_REVOLUTE:
                    self.hinge_type = 'revolute'
                elif info[2] == p.JOINT_PRISMATIC:
                    self.hinge_type = 'prismatic'
                else:
                    raise NotImplementedError('Doors must have only hinge or sliding joints')
                break

        return self.id

