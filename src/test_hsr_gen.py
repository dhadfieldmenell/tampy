import openravepy
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.robots import *

env = openravepy.Environment()
body = OpenRAVEBody(env, 'hsr', HSR())
env.SetViewer('qtcoin')
