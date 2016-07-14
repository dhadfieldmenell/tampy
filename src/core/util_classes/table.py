class Table(object):
    """
        Object stores all the information to for a table model
    """

    def __init__(self, dim):
        if isinstance(dim, str):
            dim = list(eval(dim))
        self.table_dim = [dim[0], dim[1]]
        self.thickness = dim[2]
        self.leg_dim = [dim[3], dim[4]]
        self.leg_height = dim[5]
        self.back = dim[6]
