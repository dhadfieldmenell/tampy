import internal_rep

class ParseConfigToProblem:
    """
    Read a configuration file and spawn the corresponding Problem object (see Problem class).
    """
    def __init__(self, config_file):
        self.config_file = config_file

    def parse(self):
        raise NotImplementedError
