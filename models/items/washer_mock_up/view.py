import numpy as np
import sys
sys.path.insert(0, '../../../src/')
from core.util_classes.robots import Washer
from core.util_classes.openrave_body import OpenRAVEBody
from openravepy import Environment

env = Environment()
washer = Washer()
washer_body = OpenRAVEBody(env, "washer", washer)
"""
This file is for viewing the mockup washer model in openrave simulation
"""
# env.SetViewer("qtcoin")
# import ipdb; ipdb.set_trace()
