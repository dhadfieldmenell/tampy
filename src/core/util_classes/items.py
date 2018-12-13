class Item(object):
    """
    Base class of every Objects in environment
    """
    def __init__(self):
        self._type = "item"


"""
Object defined for NAMO Domain
"""

class Circle(Item):
    """
    Defines geometry used in the NAMO domain.
    """
    def __init__(self, radius):
        self._type = "circle"
        self.radius = float(radius)
        self.col_links = set(['base'])

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

class Wall(Item):

    """
    Defines the Wall class as geometry of Obstacle used in the Namo domain.
    """

    def __init__(self, wall_type):
        self._type = "wall"
        self.wall_type = wall_type
        self.col_links = set(['base'])

"""
Obejct defined for Robot[baxter/pr2] domain
"""

class Can(Item):
    """
    Defines geometry used in the CAN domain.
    """
    def __init__(self, radius, height):
        self._type = "can"
        self.radius = float(radius)
        self.height = float(height)
        self.col_links = set(['base'])

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
        self.color = "blue"
        self._type = "sphere"
        self.radius = radius
        self.col_links = set(['base'])

class Obstacle(Item):
    """
    Defines geometry used for testing move with obstructs in the NAMO domain.
    """
    def __init__(self):
        self._type = "obstacle"
        self.col_links = set(['base'])

class Table(Obstacle):
    """
        Object stores all the information to for a table model
    """

    def __init__(self, dim):
        self._type = "table"
        if isinstance(dim, str):
            dim = list(eval(dim))
        self.table_dim = [dim[0], dim[1]]
        self.thickness = dim[2]
        self.leg_dim = [dim[3], dim[4]]
        self.leg_height = dim[5]
        self.back = dim[6]
        self.col_links = set(['base'])

class Box(Obstacle):
    """
        Object stores all the information to for a box model
    """

    def __init__(self, dim):
        self._type = "box"
        if isinstance(dim, str):
            dim = list(eval(dim))
        self.dim = dim
        self.length = dim[0]
        self.height = dim[1]
        self.width = dim[2]
        self.col_links = set(['base'])

class Basket(Item):
    """
        Object stores all the information to for a Basket model
    """

    def __init__(self):
        self._type = "basket"
        self.shape = "../models/baxter/basket.xml"
        self.up_right_rot = [0, 0, 1.57]

        self.col_links = set(['long_1', 'long_2', 'short_1', 'short_2', 'bottom'])
