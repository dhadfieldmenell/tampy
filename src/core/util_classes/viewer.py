from IPython import embed as shell
from openrave_body import OpenRAVEBody
from openravepy import Environment
from core.internal_repr.parameter import Object


class Viewer(object):
    """
    Defines viewers for visualizing execution.
    """
    def __init__(self, viewer = None):
        self.viewer = viewer

class GridWorldViewer(Viewer):
    def initialize_from_workspace(self, workspace):
        pass

class OpenRAVEViewer(Viewer):

    def __init__(self):
        self.env = Environment()
        self.env.SetViewer('qtcoin')
        self.name_to_rave_body = {}

    def initialize_from_workspace(self, workspace):
        pass

    def draw(self, objList, t):
        """
        This function draws all the objects from the objList at timestep t

        objList : list of parameters of type Object
        t       : timestep of the trajectory
        """
        for obj in objList:
            self._draw_rave_body(obj, obj.name, t)

    def draw_traj(self, objList, t_range):
        """
        This function draws the trajectory of objects from the objList

        objList : list of parameters of type Object
        t_range : range of timesteps to draw
        """
        for t in t_range:
            for obj in objList:
                name = "{0}-{1}".format(obj.name, t)
                self._draw_rave_body(obj, name, t)

    def _draw_rave_body(self, obj, name, t):
        assert isinstance(obj, Object)
        if name not in self.name_to_rave_body:
            self.name_to_rave_body[name] = OpenRAVEBody(self.env, name, obj.geom)
        self.name_to_rave_body[name].set_pose(obj.pose[:, t])
