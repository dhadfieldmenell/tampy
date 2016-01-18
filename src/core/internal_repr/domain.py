from IPython import embed as shell

class Domain(object):
    """
    A single Domain object gets created every time p_mod_abs is run. It stores the parameter, predicate, and action schemas.

    parameter schema:
    {parameter type : (Object or Symbol class, {attribute dictionary}}
    where attribute dictionary maps each attribute name to the class that attribute is an object of, e.g.
    {"name": str, "pose": Vector2d}

    predicate schema:
    {predicate type : (predicate class, [list of expected parameter types as strings])}

    action schema:
    TODO
    """
    def __init__(self, param_schema, pred_schema, action_schema):
        self.param_schema = param_schema
        self.pred_schema = pred_schema
        self.action_schema = action_schema
