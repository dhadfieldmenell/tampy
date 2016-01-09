class ParseConfigToSolvers:
    """
    Read a configuration file and spawn HLSolver and LLSolver objects.
    """
    def __init__(self, config_file):
        self.config_file = config_file

    def parse(self):
        raise NotImplementedError
