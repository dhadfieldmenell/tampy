class Predicate:
    """
    Predicates hold a set of parameters (see Parameter class) and represent testable relationships among
    these parameters. The test occurs for a particular time slice.
    """
    def __init__(self, name, params):
        self.name = name
        self.params = params

    def test(self, start_time, end_time):
        raise NotImplementedError
