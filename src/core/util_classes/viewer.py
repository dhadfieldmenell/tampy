from software_constants import *
from .openrave_body import OpenRAVEBody
if USE_ROS:
    from openravepy import Environment
from core.internal_repr.parameter import Object
from core.util_classes.robots import Robot, PR2, Baxter, Washer, HSR
from core.util_classes.items import Can, Table, Box
import numpy as np
import time, os, os.path as osp, shutil, scipy.misc, subprocess
import pybullet as P


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
    _viewer = None

    def __init__(self, env = None):
        assert OpenRAVEViewer._viewer == None
        if env == None:
            self.env = Environment()
        else:
            self.env = env
        self.env.SetViewer('qtcoin')
        self.name_to_rave_body = {}
        OpenRAVEViewer._viewer = self

    def clear(self):
        for b in self.name_to_rave_body.values():
            b.delete()
        self.name_to_rave_body = {}

    @staticmethod
    def create_viewer(env = None):
        # if reset and OpenRAVEViewer._viewer != None:
        #     ## close the old viewer to avoid a threading error
        #     OpenRAVEViewer._viewer = None
        if OpenRAVEViewer._viewer == None:
            return OpenRAVEViewer(env)
        OpenRAVEViewer._viewer.clear()
        return OpenRAVEViewer._viewer

    def lazy_spawn_or_body(self, param, name, geom):
        if param.openrave_body is not None:
            assert geom == param.openrave_body._geom
            assert self.env == param.openrave_body.env_body.GetEnv()
        else:
            param.openrave_body = OpenRAVEBody(self.env, name, geom)
        return param.openrave_body

    def record_plan(self, plan, outf, res = (640, 480), cam=None):
        """
        creates a video of a plan and stores it in outf
        """
        obj_list = []
        horizon = plan.horizon
        v = self.env.GetViewer()
        if osp.exists('.video_images'):
            shutil.rmtree('.video_images')
        os.makedirs('.video_images')
        for p in plan.params.values():
            if not p.is_symbol():
                obj_list.append(p)
        for t in range(horizon):
            self.draw(obj_list, t)
            time.sleep(1)
            if cam is None:
                cam = v.GetCameraTransform()
            v.SendCommand('SetFiguresInCamera 1')
            im = v.GetCameraImage(res[0], res[1], cam,[640,640,320,240])
            scipy.misc.imsave('.video_images/frame_'+str('%05d'%t)+'.png', im)

        outfname = "{}.mp4".format(outf)
        if osp.exists(outfname):
            os.remove(outfname)
        arglist = ['avconv',
                   '-f', 'image2',
                   '-r', '10',
                   "-i", ".video_images/frame_%05d.png",
                   "-f", "mp4",
                   "-bf", "1",
                   "-r", "30",
                   "{}.mp4".format(outf)]
        subprocess.call(arglist)


    def initialize_from_workspace(self, workspace):
        pass

    def draw(self, objList, t, transparency = 0.7):
        """
        This function draws all the objects from the objList at timestep t

        objList : list of parameters of type Object
        t       : timestep of the trajectory
        """
        for obj in objList:
            self._draw_rave_body(obj, obj.name, t, transparency)

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

    def _draw_rave_body(self, obj, name, t, transparency = 0.7):
        rotation = np.array([[0],[0],[0]])
        pose = obj.pose[:,t]

        assert isinstance(obj, Object) and not np.any(np.isnan(pose))
        if name not in self.name_to_rave_body:
            self.name_to_rave_body[name] = OpenRAVEBody(self.env, name, obj.geom)
            obj.openrave_body = self.name_to_rave_body[name]

        if isinstance(obj.geom, Robot):
            dof_value_map = None
            pose = [0, 0, pose]
            if isinstance(obj.geom, PR2):
                dof_value_map = {"backHeight": obj.backHeight[:, t],
                                 "lArmPose": obj.lArmPose[:, t],
                                 "lGripper": obj.lGripper[:, t],
                                 "rArmPose": obj.rArmPose[:, t],
                                 "rGripper": obj.rGripper[:, t]}
            elif isinstance(obj.geom, Baxter):
                dof_value_map = {"lArmPose": obj.lArmPose[:, t],
                                 "lGripper": obj.lGripper[:, t],
                                 "rArmPose": obj.rArmPose[:, t],
                                 "rGripper": obj.rGripper[:, t]}
            elif isinstance(obj.geom, HSR):
                dof_value_map = {"arm": obj.arm[:, t],
                                 "gripper": obj.gripper[:, t]}
                pose = obj.pose[:, t]
            elif isinstance(obj.geom, Washer):
                dof_value_map = {"door": obj.door[:, t]}
                pose = obj.pose[:, t]
                rotation = obj.rotation[:, t]

            self.name_to_rave_body[name].set_dof(dof_value_map)
        else:
            if hasattr(obj, 'rotation'):
                rotation = obj.rotation[:, t]
            else:
                rotation = [0, 0, 0]
            assert not np.any(np.isnan(rotation))

        self.name_to_rave_body[name].set_pose(pose, rotation)
        self.name_to_rave_body[name].set_transparency(transparency)

    def animate_plan(self, plan, delay=.1):
        self.animate_range(plan, (0, plan.horizon-1), delay=delay)

    def animate_range(self, plan, xxx_todo_changeme, delay=.1):
        (start, end) = xxx_todo_changeme
        obj_list = self._get_plan_obj_list(plan)
        for t in range(start, end+1):
            self.draw(obj_list, t)
            time.sleep(delay)

    def draw_plan(self, plan):
        horizon = plan.horizon
        self.draw_plan_range(plan, (0, horizon-1))

    def draw_plan_range(self, plan, xxx_todo_changeme1):
        (start, end) = xxx_todo_changeme1
        obj_list = self._get_plan_obj_list(plan)
        self.draw_traj(obj_list, list(range(start, end+1)))

    def _get_plan_obj_list(self, plan):
        obj_list = []
        for p in plan.params.values():
            if not p.is_symbol():
                obj_list.append(p)
        return obj_list


    def draw_plan_ts(self, plan, t):
        obj_list = []
        horizon = plan.horizon
        for p in plan.params.values():
            if not p.is_symbol():
                obj_list.append(p)
        self.draw(obj_list, t)

    def draw_cols(self, plan):
        horizon = plan.horizon
        for t in range(horizon):
            self.draw_cols_ts(plan, t)

    def draw_cols_range(self, plan, xxx_todo_changeme2):
        (start, end) = xxx_todo_changeme2
        for t in range(start, end+1):
            self.draw_cols_ts(plan, t)

    def draw_cols_ts(self, plan, t):
        preds = plan.get_active_preds(t)
        for p in preds:
            try:
                p.plot_cols(self.env, t)
            except AttributeError:
                ## some predicates won't define a collision
                continue

class PyBulletViewer(Viewer):
    def __init__(self, envid=None, visual=None):
        if envid is None:
            # If a visual doesn't yet exist, make one by checking the Display
            if visual is None:
                visual = len(os.environ.get('DISPLAY', '')) > 0
            if not P.getConnectionInfo()['isConnected']:
                server = P.GUI if visual else P.DIRECT
                envid = P.connect(server)
                P.resetSimulation()

        self.visual = visual
        self.env = envid
        self.name_to_rave_body = {}
        PyBulletViewer._viewer = self

    def clear(self):
        for b in self.name_to_rave_body.values():
            b.delete()
        self.name_to_rave_body = {}

    @staticmethod
    def create_viewer(env = None):
        # if reset and PybulletViewer._viewer != None:
        #     ## close the old viewer to avoid a threading error
        #     PybulletViewer._viewer = None
        if PyBulletViewer._viewer == None:
            return PyBulletViewer(env)
        PyBulletViewer._viewer.clear()
        return PyBulletViewer._viewer

    def lazy_spawn_or_body(self, param, name, geom):
        if param.openrave_body is None:
            param.openrave_body = OpenRAVEBody(self.env, name, geom)
        return param.openrave_body

    def initialize_from_workspace(self, workspace):
        pass

    def draw(self, objList, t, transparency = 0.7):
        """
        This function draws all the objects from the objList at timestep t

        objList : list of parameters of type Object
        t       : timestep of the trajectory
        """
        for obj in objList:
            self._draw_rave_body(obj, obj.name, t, transparency)

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

    def _draw_rave_body(self, obj, name, t, transparency = 0.7):
        rotation = np.array([[0],[0],[0]])
        pose = obj.pose[:,t]

        assert isinstance(obj, Object) and not np.any(np.isnan(pose))
        if name not in self.name_to_rave_body:
            self.name_to_rave_body[name] = OpenRAVEBody(self.env, name, obj.geom)
            obj.openrave_body = self.name_to_rave_body[name]

        if isinstance(obj.geom, Robot):
            dof_value_map = None
            pose = [0, 0, pose]
            if isinstance(obj.geom, PR2):
                dof_value_map = {"backHeight": obj.backHeight[:, t],
                                 "lArmPose": obj.lArmPose[:, t],
                                 "lGripper": obj.lGripper[:, t],
                                 "rArmPose": obj.rArmPose[:, t],
                                 "rGripper": obj.rGripper[:, t]}
            elif isinstance(obj.geom, Baxter):
                dof_value_map = {"lArmPose": obj.lArmPose[:, t],
                                 "lGripper": obj.lGripper[:, t],
                                 "rArmPose": obj.rArmPose[:, t],
                                 "rGripper": obj.rGripper[:, t]}
            elif isinstance(obj.geom, HSR):
                dof_value_map = {"arm": obj.arm[:, t],
                                 "gripper": obj.gripper[:, t]}
                pose = obj.pose[:, t]
            elif isinstance(obj.geom, Washer):
                dof_value_map = {"door": obj.door[:, t]}
                pose = obj.pose[:, t]
                rotation = obj.rotation[:, t]

            self.name_to_rave_body[name].set_dof(dof_value_map)
        else:
            if hasattr(obj, 'rotation'):
                rotation = obj.rotation[:, t]
            else:
                rotation = [0, 0, 0]
            assert not np.any(np.isnan(rotation))

        self.name_to_rave_body[name].set_pose(pose, rotation)
        self.name_to_rave_body[name].set_transparency(transparency)

    def animate_plan(self, plan, delay=.1):
        self.animate_range(plan, (0, plan.horizon-1), delay=delay)

    def animate_range(self, plan, xxx_todo_changeme, delay=.1):
        (start, end) = xxx_todo_changeme
        obj_list = self._get_plan_obj_list(plan)
        for t in range(start, end+1):
            self.draw(obj_list, t)
            time.sleep(delay)

    def draw_plan(self, plan):
        horizon = plan.horizon
        self.draw_plan_range(plan, (0, horizon-1))

    def draw_plan_range(self, plan, xxx_todo_changeme1):
        (start, end) = xxx_todo_changeme1
        obj_list = self._get_plan_obj_list(plan)
        self.draw_traj(obj_list, list(range(start, end+1)))

    def _get_plan_obj_list(self, plan):
        obj_list = []
        for p in plan.params.values():
            if not p.is_symbol():
                obj_list.append(p)
        return obj_list


    def draw_plan_ts(self, plan, t):
        obj_list = []
        horizon = plan.horizon
        for p in plan.params.values():
            if not p.is_symbol():
                obj_list.append(p)
        self.draw(obj_list, t)

    def draw_cols(self, plan):
        horizon = plan.horizon
        for t in range(horizon):
            self.draw_cols_ts(plan, t)

    def draw_cols_range(self, plan, xxx_todo_changeme2):
        (start, end) = xxx_todo_changeme2
        for t in range(start, end+1):
            self.draw_cols_ts(plan, t)

    def draw_cols_ts(self, plan, t):
        preds = plan.get_active_preds(t)
        for p in preds:
            try:
                p.plot_cols(self.env, t)
            except AttributeError:
                ## some predicates won't define a collision
                continue

