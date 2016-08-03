class Box(object):
    """
        Object stores all the information to for a box model
    """

    def __init__(self, dim):
        if isinstance(dim, str):
            dim = list(eval(dim))
        self.dim = dim
        self.length = dim[0]
        self.height = dim[1]
        self.width = dim[2]
