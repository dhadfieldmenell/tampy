class Can(object):
    """
    Defines geometry used in the CAN domain.
    """
    def __init__(self, radius, height):
        self.radius = float(radius)
        self.height = float(height)

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
        super(RedCan, self).__init__(radius, height)
        self.color = "green"
