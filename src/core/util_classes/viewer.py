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
        self.param_to_rave_body = {}

    def initialize_from_workspace(self, workspace):
        pass

    def draw(self, objList, t):
        """
        This function draws all the objects from the objList at timestep t

        objList : list of parameters of type Object
        t       : timestep of the trajectory
        """
        for obj in objList:
            assert isinstance(obj, Object)
            if obj not in self.param_to_rave_body:
                self.param_to_rave_body[obj] = OpenRAVEBody(self.env, obj.name, obj.geom)
            self.param_to_rave_body[obj].set_pose(obj.pose[:, t])

    def draw_traj(self, objList, t_range):
        """
        This function draws the trajectory of objects from the objList

        objList : list of parameters of type Object
        t_range : range of timesteps to draw
        """
        for t in t_range:
            for obj in objList:
                rave_body = OpenRAVEBody(self.env, "{0}-{1}".format(obj.name, t), obj.geom)
                rave_body.set_pose(obj.pose[:, t])
