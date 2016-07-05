from IPython import embed as shell

class Viewer(object):
    """
    Defines viewers for visualizing execution.
    """
    def __init__(self, viewer):
        self.viewer = viewer

class GridWorldViewer(Viewer):
    def initialize_from_workspace(self, workspace):
        pass

class OpenRAVEViewer(Viewer):
    def initialize_from_workspace(self, workspace):
        pass