import numpy as np
import os
import pybullet as P
from core.util_classes.robots import Baxter
from core.util_classes.openrave_body import *
from core.util_classes.transform_utils import *
from core.util_classes.viewer import PyBulletViewer


server = P.GUI if len(os.environ.get('DISPLAY', '')) else P.DIRECT
P.connect(server)
robot = Baxter()
body = OpenRAVEBody(None, 'baxter', robot)
pos = [0.5, 0.5, 0.3]
quat = OpenRAVEBody.quat_from_v1_to_v2([0,0,1], [0,0,-1])
iks = body.get_ik_from_pose(pos, quat, 'left', multiple=True)
import ipdb; ipdb.set_trace()

