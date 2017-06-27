import numpy as np
import sys
sys.path.insert(0, '../../../src/')
from core.util_classes.robots import Washer
from core.util_classes.openrave_body import OpenRAVEBody
from openravepy import Environment, KinBody, RaveCreateRobot

env = Environment()
washer = Washer()
washer_body = OpenRAVEBody(env, "washer", washer)

# external1 = OpenRAVEBody.create_body_info(KinBody.Link.GeomType.Box, [0.025,0.40,0.275], [1, 0.55, 0.31])
# external1._fTransparency = 0.3
# external1._t = OpenRAVEBody.transform_from_obj_pose([-0.3,0,0],[0,0,0])
# external1_body = RaveCreateRobot(env, '')
# external1_body.InitFromGeometries([external1])
# external1_body.SetName("external1")
# env.Add(external1_body)
# def set_transparency(body, transparency):
#         for link in body.GetLinks():
#             for geom in link.GetGeometries():
#                 geom.SetTransparency(transparency)

# env_body = env.ReadKinBodyXMLFile("washer_col.xml")
# env.Add(env_body)
# set_transparency(env_body, 0.3)
# env_body.SetDOFValues([-np.pi/2])
env_body = washer_body.env_body
"""
This file is for viewing the mockup washer model in openrave simulation
"""
env.SetViewer("qtcoin")
import ipdb; ipdb.set_trace()
