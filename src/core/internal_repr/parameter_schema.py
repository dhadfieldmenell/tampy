class ParameterSchema(object):
    """
    A parameter schema holds the following information.

    param_type: type of this parameter
    param_class: reference to either Object or Symbol class
    attr_dict: maps each attribute name to the class that attribute is an object of, e.g. {"name": str, "pose": Vector2d}
    """
    def __init__(self, param_type, param_class, attr_dict, types=[]):
        self.param_type = param_type
        self.param_class = param_class
        self.attr_dict = attr_dict
        self.types = types if len(types) else [param_type]
