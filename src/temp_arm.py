from core.util_classes.robots import TwoLinkArm
from core.util_classes.openrave_body import *

import pybullet as P

env = P.connect(P.DIRECT)
body = OpenRAVEBody(env, 'arm', TwoLinkArm())
import ipdb; ipdb.set_trace()

