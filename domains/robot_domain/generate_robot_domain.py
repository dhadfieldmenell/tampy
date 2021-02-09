import sys
sys.path.insert(0, '../../src/')
import core.util_classes.common_constants as const
from core.util_classes.robots import *

dom_str = """
# AUTOGENERATED. DO NOT EDIT.
# Configuration file for CAN domain. Blank lines and lines beginning with # are filtered out.

# implicity, all types require a name
Types: Robot, RobotPose, CollisionShape 

# Define the class location of each non-standard attribute type used in the above parameter type descriptions.

Attribute Import Paths: Baxter core.util_classes.robots, Vector1d core.util_classes.matrix, Vector3d core.util_classes.matrix, ArmPose7d core.util_classes.matrix, Table core.util_classes.items, Box core.util_classes.items, Basket core.util_classes.items, Cloth core.util_classes.items

Predicates Import Path: core.util_classes.robot_predicates

"""

# Automated handling to setup robot types
robots = ['Baxter']
r_types = ""
for r in robots:
    r_types += "{}, ".format(r)
r_types = r_types[:-2] + " - Robot"

rpose_types = ""
for r in robots:
    rpose_types += "{}Pose, ".format(r)
rpose_types = rpose_types[:-2] + " - RobotPose"

subtypes = "\nSubtypes: Obstacle, Reachable - CollisionShape; Item, Target - Reachable; Cloth, Can, Basket - Item; ClothTarget, CanTarget, BasketTarget - Target"
subtypes += "; " + r_types + "; " + rpose_types + "\n"
dom_str += subtypes + "\n"

class PrimitivePredicates(object):
    def __init__(self):
        self.attr_dict = {}

    def add(self, name, attrs):
        self.attr_dict[name] = attrs

    def get_str(self):
        prim_str = 'Primitive Predicates: '
        first = True
        for name, attrs in list(self.attr_dict.items()):
            for attr_name, attr_type in attrs:
                pred_str = attr_name + ', ' + name + ', ' + attr_type
                if first:
                    prim_str += pred_str
                    first = False
                else:
                    prim_str += '; ' + pred_str
        return prim_str

pp = PrimitivePredicates()
pp.add('Reachable', [('value', 'Vector3d'), ('rotation', 'Vector3d')])
pp.add('CollisionShape', [('pose', 'Vector3d'), ('rotation', 'Vector3d')])
pp.add('Item', [('pose', 'Vector3d'), ('rotation', 'Vector3d')])
pp.add('Target', [('value', 'Vector3d'), ('rotation', 'Vector3d')])

pp.add('Basket', [('geom', 'Basket'), ('pose', 'Vector3d'), ('rotation', 'Vector3d')])
pp.add('BasketTarget', [('geom', 'Basket'), ('value', 'Vector3d'), ('rotation', 'Vector3d')])
pp.add('Cloth', [('geom', 'Cloth'), ('pose', 'Vector3d'), ('rotation', 'Vector3d')])
pp.add('ClothTarget', [('value', 'Vector3d'), ('rotation', 'Vector3d')])
pp.add('RobotPose', [('value', 'Vector3d')])
pp.add('Robot', [('pose', 'Vector3d')])
pp.add('Obstacle', [('geom', 'Box'), ('pose', 'Vector3d'), ('rotation', 'Vector3d')])

for r in robots:
    try:
        r_geom = eval("{0}()".format(r))
    except:
        print('Could not load geom for {}'.format(r))
        continue
    attrs = [('geom', r)]
    pose_attrs = []
    for arm in r_geom.arms:
        njnts = len(r_geom.jnt_names[arm])
        attrs.append((arm, 'ArmPose{0}d'.format(njnts)))
        pose_attrs.append((arm, 'ArmPose{0}d'.format(njnts)))
        gripper = r_geom.get_gripper(arm)
        attrs.append((gripper, 'Vector1d'))
        pose_attrs.append((gripper, 'Vector1d'))
        attrs.append(('{}_ee_pos'.format(arm), 'Vector3d'))
        pose_attrs.append(('{}_ee_pos'.format(arm), 'Vector3d'))
    pp.add(r, attrs)
    pp.add(r+"Pose", pose_attrs)

dom_str += pp.get_str() + '\n\n'

class DerivatedPredicates(object):
    def __init__(self):
        self.pred_dict = {}

    def add(self, name, args):
        self.pred_dict[name] = args

    def get_str(self):
        prim_str = 'Derived Predicates: '

        first = True
        for name, args in list(self.pred_dict.items()):
            pred_str = name
            for arg in args:
                pred_str += ', ' + arg

            if first:
                prim_str += pred_str
                first = False
            else:
                prim_str += '; ' + pred_str
        return prim_str

dp = DerivatedPredicates()
dp.add('At', ['Item', 'Target'])
dp.add('Near', ['Item', 'Target'])
dp.add('RobotAt', ['Robot', 'RobotPose'])
dp.add('IsMP', ['Robot'])
dp.add('WithinJointLimit', ['Robot'])
dp.add('Stationary', ['Item'])
dp.add('StationaryNEq', ['Item', 'Item'])
dp.add('StationaryBase', ['Robot'])
dp.add('StationaryArms', ['Robot'])
dp.add('StationaryLeftArm', ['Robot'])
dp.add('StationaryRightArm', ['Robot'])
dp.add('StationaryW', ['Obstacle'])
dp.add('CloseGripperLeft', ['Robot'])
dp.add('CloseGripperRight', ['Robot'])
dp.add('OpenGripperLeft', ['Robot'])
dp.add('OpenGripperRight', ['Robot'])
dp.add('CloseGripper', ['Robot'])
dp.add('OpenGripper', ['Robot'])
dp.add('Obstructs', ['Robot', 'RobotPose', 'RobotPose', 'CollisionShape'])
dp.add('ObstructsHolding', ['Robot', 'RobotPose', 'RobotPose', 'CollisionShape', 'CollisionShape'])
dp.add('Collides', ['CollisionShape', 'CollisionShape'])
dp.add('RCollides', ['Robot', 'CollisionShape'])
dp.add('RSelfCollides', ['Robot'])
dp.add('EEReachableLeft', ['Robot', 'Reachable'])
dp.add('EEReachableRight', ['Robot', 'Reachable'])
dp.add('ApproachLeft', ['Robot', 'Reachable'])
dp.add('ApproachRight', ['Robot', 'Reachable'])
dp.add('NearApproachLeft', ['Robot', 'Reachable'])
dp.add('NearApproachRight', ['Robot', 'Reachable'])
dp.add('InGripperLeft', ['Robot', 'Item'])
dp.add('InGripperRight', ['Robot', 'Item'])
dp.add('InGripper', ['Robot', 'Item'])
dp.add('NearGripperLeft', ['Robot', 'Item'])
dp.add('NearGripperRight', ['Robot', 'Item'])
dp.add('GripperAtLeft', ['Robot', 'Item'])
dp.add('GripperAtRight', ['Robot', 'Item'])
dp.add('GripperAt', ['Robot', 'Item'])
dp.add('GrippersDownRot', ['Robot'])
dp.add('LeftGripperDownRot', ['Robot'])
dp.add('RightGripperDownRot', ['Robot'])
dp.add('EEValid', ['Robot'])
dp.add('LeftEEValid', ['Robot'])
dp.add('RightEEValid', ['Robot'])

dom_str += dp.get_str() + '\n'

dom_str += """

# The first set of parentheses after the colon contains the
# parameters. The second contains preconditions and the third contains
# effects. This split between preconditions and effects is only used
# for task planning purposes. Our system treats all predicates
# similarly, using the numbers at the end, which specify active
# timesteps during which each predicate must hold

"""

class Action(object):
    def __init__(self, name, timesteps, pre=None, post=None):
        pass

    def to_str(self):
        time_str = ''
        cond_str = '(and '
        for pre, timesteps in self.pre:
            cond_str += pre + ' '
            time_str += timesteps + ' '
        cond_str += ')'

        cond_str += '(and '
        for eff, timesteps in self.eff:
            cond_str += eff + ' '
            time_str += timesteps + ' '
        cond_str += ')'

        return "Action " + self.name + ' ' + str(self.timesteps) + ': ' + self.args + ' ' + cond_str + ' ' + time_str

class Move(Action):
    def __init__(self):
        self.name = 'moveto'
        self.timesteps = 15 # 25
        end = self.timesteps - 1
        self.end = end
        self.args = '(?robot - Robot ?start - RobotPose ?end - RobotPose)'
        self.pre = [\
            ('(RobotAt ?robot ?start)', '{}:{}'.format(0, -1)),
            ('(not (RobotAt ?robot ?end))', '{}:{}'.format(0, -1)),
            ('(forall (?obj - Item)\
                (not (Obstructs ?robot ?start ?end ?obj)))', '{}:{}'.format(1, end-1)),
            ('(forall (?obj - Item)\
                (Stationary ?obj))', '{}:{}'.format(0, end-1)),
            ('(forall (?obs - Obstacle) (StationaryW ?obs))', '{}:{}'.format(0, end-1)),
            ('(IsMP ?robot)', '{}:{}'.format(0, end-1)),
            #('(LeftEEValid ?robot)', '{}:{}'.format(1, end-1)),
            #('(RightEEValid ?robot)', '{}:{}'.format(1, end-1)),
            ('(WithinJointLimit ?robot)', '{}:{}'.format(0, end)),
            ('(forall (?w - Obstacle) (not (RCollides ?robot ?w)))', '{}:{}'.format(1, end-1)),
            # ('(not (RSelfCollides ?robot))', '0:{}'.format(end)),
        ]
        self.eff = [\
            (' (not (RobotAt ?robot ?start))', '{}:{}'.format(end, end-1)),
            ('(RobotAt ?robot ?end)', '{}:{}'.format(end, end-1))]


class MoveLeft(Move):
    def __init__(self):
        super(MoveLeft, self).__init__()
        self.name = 'move_to_left'
        self.pre.extend([
            ('(forall (?obj - Item)\
                (not (InGripperLeft ?robot ?obj)))', '{}:{}'.format(0, 0)),
            ('(OpenGripperLeft ?robot)', '1:{}'.format(self.end-1)),
            ('(StationaryRightArm ?robot)', '0:{}'.format(self.end-1))])


class MoveRight(Move):
    def __init__(self):
        super(MoveLeft, self).__init__()
        self.name = 'move_to_right'
        self.pre.extend([
            ('(forall (?obj - Item)\
                (not (InGripperRight ?robot ?obj)))', '{}:{}'.format(0, 0)),
            ('(OpenGripperRight ?robot)', '1:{}'.format(self.end-1)),
            ('(StationaryLeftArm ?robot)', '0:{}'.format(self.end-1))])


class MoveToGraspLeft(MoveLeft):
    def __init__(self):
        super(MoveToGraspLeft, self).__init__()
        self.name = 'move_to_grasp_left'
        self.args = '(?robot - Robot ?item - Item ?targ - Target ?start - RobotPose ?end - RobotPose)'
        self.pre.extend([('(At ?item ?targ)', '0:0'), ('(At ?item ?targ)', '{0}:{1}'.format(self.end-1, self.end))])
        self.eff.extend([('(ApproachLeft ?robot ?item)', '{0}:{1}'.format(self.end, self.end-1)),
                         ('(NearApproachLeft ?robot ?item)', '{0}:{1}'.format(self.end, self.end)),
            ('(forall (?obj - Reachable / ?item) (not (ApproachLeft ?robot ?obj)))', '{0}:{1}'.format(self.end, self.end-1)),
            ('(forall (?obj - Reachable / ?item) (not (NearApproachLeft ?robot ?obj)))', '{0}:{1}'.format(self.end, self.end-1))])


class MoveToGraspRight(MoveRight):
    def __init__(self):
        super(MoveToGraspRight, self).__init__()
        self.name = 'move_to_grasp_right'
        self.args = '(?robot - Robot ?item - Item ?targ - Target ?start - RobotPose ?end - RobotPose)'
        self.pre.extend([('(At ?item ?targ)', '0:0'), ('(At ?item ?targ)', '{0}:{1}'.format(self.end-1, self.end))])
        self.eff.extend([('(ApproachRight ?robot ?item)', '{0}:{1}'.format(self.end, self.end-1)),
                         ('(NearApproachRight ?robot ?item)', '{0}:{1}'.format(self.end, self.end)),
            ('(forall (?obj - Reachable / ?item) (not (ApproachRight ?robot ?obj)))', '{0}:{1}'.format(self.end, self.end-1)),
            ('(forall (?obj - Reachable / ?item) (not (NearApproachRight ?robot ?obj)))', '{0}:{1}'.format(self.end, self.end-1))])


class MoveHolding(Action):
    def __init__(self):
        self.name = 'moveholding'
        self.timesteps = 15 # 25
        end = self.timesteps - 1
        self.end = end
        self.args = '(?robot - Robot ?start - RobotPose ?end - RobotPose ?item - Item)'
        self.pre = [\
            ('(RobotAt ?robot ?start)', '0:-1'),
            ('(not (RobotAt ?robot ?end))', '{}:{}'.format(0, -1)),
            ('(forall (?obj - Item)\
                (not (ObstructsHolding ?robot ?start ?end ?obj ?item))\
            )', '1:{}'.format(end)),
            ('(forall (?obj - Item)\
                (StationaryNEq ?obj ?item))', '{}:{}'.format(0, end-1)),
            ('(forall (?obs - Obstacle) (StationaryW ?obs))', '0:{}'.format(end-1)),
            #('(LeftEEValid ?robot)', '{}:{}'.format(1, end-1)),
            #('(RightEEValid ?robot)', '{}:{}'.format(1, end-1)),
            ('(IsMP ?robot)', '0:{}'.format(end-1)),
            ('(WithinJointLimit ?robot)', '0:{}'.format(end)),
            # ('(not (RSelfCollides ?robot))', '0:{}'.format(end)),
            ('(forall (?obs - Obstacle) (not (RCollides ?robot ?obs)))', '1:{}'.format(end-1))
        ]
        self.eff = [\
            ('(not (RobotAt ?robot ?start))', '{}:{}'.format(end, end-1)),
            ('(RobotAt ?robot ?end)', '{}:{}'.format(end, end-1))
        ]


class MoveHoldingLeft(MoveHolding):
    def __init__(self):
        super(MoveHoldingLeft, self).__init__()
        self.pre.extend([
            ('(CloseGripperLeft ?robot)', '1:{}'.format(self.end-1)),
            ('(InGripperLeft ?robot ?item)', '{0}:{1}'.format(0, -1)),
            ('(NearGripperLeft ?robot ?item)', '{0}:{1}'.format(0, 0)),
            ('(InGripperLeft ?robot ?item)', '{}:{}'.format(self.end-1, self.end)),
            ('(NearGripperLeft ?robot ?item)', '{}:{}'.format(self.end, self.end)),
            ('(StationaryRightArm ?robot)', '0:{}'.format(self.end-1))])


class MoveHoldingRight(MoveHolding):
    def __init__(self):
        super(MoveHoldingRight, self).__init__()
        self.pre.extend([
            ('(CloseGripperRight ?robot)', '1:{}'.format(self.end-1)),
            ('(InGripperRight ?robot ?item)', '{0}:{1}'.format(0, -1)),
            ('(NearGripperRight ?robot ?item)', '{0}:{1}'.format(0, 0)),
            ('(InGripperRight ?robot ?item)', '{0}:{1}'.format(self.end-1, self.end)),
            ('(NearGripperRight ?robot ?item)', '{0}:{1}'.format(self.end, self.end)),
            # ('(InGripperRight ?robot ?item)', '{}:{}'.format(self.end-1)),
            ('(StationaryLeftArm ?robot)', '0:{}'.format(self.end-1))])


class MoveToPutdownLeft(MoveHoldingLeft):
    def __init__(self):
        super(MoveToPutdownLeft, self).__init__()
        self.name = 'move_to_putdown_left'
        self.args = '(?robot - Robot ?targ - Target ?item - Item ?start - RobotPose ?end - RobotPose)'
        self.pre.extend([('(forall (?obj - Item) (not (At ?obj ?targ)))', '0:0')])
        self.eff.extend([('(ApproachLeft ?robot ?targ)', '{0}:{1}'.format(self.end, self.end-1)),
                         ('(NearApproachLeft ?robot ?targ)', '{0}:{1}'.format(self.end, self.end))])


class MoveToPutdownRight(MoveHoldingRight):
    def __init__(self):
        super(MoveToPutdownRight, self).__init__()
        self.name = 'move_to_putdown_right'
        self.args = '(?robot - Robot ?targ - Target ?item - Item ?start - RobotPose ?end - RobotPose)'
        self.pre.extend([('(forall (?obj - Item) (not (At ?obj ?targ)))', '0:0')])
        self.eff.extend([('(ApproachRight ?robot ?targ)', '{0}:{1}'.format(self.end, self.end-1)),
                         ('(NearApproachRight ?robot ?targ)', '{0}:{1}'.format(self.end, self.end))])



class Grasp(Action):
    def __init__(self):
        self.name = 'grasp'
        self.timesteps = 5 + 2 * const.EEREACHABLE_STEPS # 2 * const.EEREACHABLE_STEPS + 11
        end = self.timesteps - 1
        self.end = end
        self.args = '(?robot - Robot ?item - Item ?target - Target ?sp - RobotPose ?ep - RobotPose)'
        grasp_time = end // 2 # const.EEREACHABLE_STEPS + 5
        self.grasp_time = grasp_time
        self.pre = [\
            ('(At ?item ?target)', '0:0'),
            ('(At ?item ?target)', '1:{}'.format(grasp_time)),
            ('(RobotAt ?robot ?sp)', '0:-1'),
            ('(not (RobotAt ?robot ?ep))', '{}:{}'.format(0, 0)),
            #('(LeftEEValid ?robot)', '{}:{}'.format(1, end-1)),
            #('(RightEEValid ?robot)', '{}:{}'.format(1, end-1)),
            ('(forall (?obj - Item) \
                (Stationary ?obj)\
            )', '0:{}'.format(grasp_time-1)),
            ('(forall (?obj - Item) \
                (StationaryNEq ?obj ?item)\
            )', '0:{}'.format(end-1)),
            ('(forall (?obs - Obstacle)\
                (StationaryW ?obs)\
            )', '{}:{}'.format(0, end-1)),
            ('(IsMP ?robot)', '0:{}'.format(end-1)),
            ('(WithinJointLimit ?robot)', '0:{}'.format(end)),
            #('(forall (?obs - Obstacle)\
            #    (not (Collides ?item ?obs))\
            #)', '1:{}'.format(end-1)),
            ('(forall (?obs - Obstacle)\
                (not (RCollides ?robot ?obs))\
            )', '1:{}'.format(self.grasp_time-2)),
            ('(forall (?obs - Obstacle)\
                (not (RCollides ?robot ?obs))\
            )', '{0}:{1}'.format(self.grasp_time+1, self.end-1)),
            ('(forall (?obj - Item)\
                (not (Obstructs ?robot ?sp ?ep ?obj))\
            )', '1:{}'.format(grasp_time-3)),
            ('(forall (?obj - Item)\
                (not (ObstructsHolding ?robot ?sp ?ep ?obj ?item))\
            )', '{}:{}'.format(grasp_time-1, end-1))
        ]
        self.eff = [\
            ('(not (At ?item ?target))', '{}:{}'.format(end, end-1)) ,
            ('(not (RobotAt ?robot ?sp))', '{}:{}'.format(end, end-1)),
            ('(RobotAt ?robot ?ep)', '{}:{}'.format(end, end-1)),
            ('(forall (?sym1 - RobotPose)\
                (forall (?sym2 - RobotPose)\
                    (not (Obstructs ?robot ?sym1 ?sym2 ?item))\
                )\
            )', '{}:{}'.format(end, end-1)),
            ('(forall (?sym1 - Robotpose)\
                (forall (?sym2 - RobotPose)\
                    (forall (?obj - Item) (not (ObstructsHolding ?robot ?sym1 ?sym2 ?item ?obj)))\
                )\
            )', '{}:{}'.format(end, end-1))
        ]


class GraspLeft(Grasp):
    def __init__(self):
        super(GraspLeft, self).__init__()
        steps = const.EEREACHABLE_STEPS
        self.name = 'grasp_left'
        self.pre.extend([
            ('(NearApproachLeft ?robot ?item)', '0:0'),
            #('(ApproachLeft ?robot ?item)', '0:0'),
            ('(not (InGripperRight ?robot ?item))', '0:0'),
            ('(not (InGripperLeft ?robot ?item))', '0:0'),
            ('(EEReachableLeft ?robot ?item)', '{}:{}'.format(self.grasp_time, self.grasp_time)),
            ('(OpenGripperLeft ?robot)', '{}:{}'.format(1,  self.grasp_time-1)),
            ('(CloseGripperLeft ?robot)', '{}:{}'.format(self.grasp_time,  self.end-1)),
            ('(InGripperLeft ?robot ?item)', '{}:{}'.format(self.grasp_time, self.grasp_time)),
            #('(LeftGripperDownRot ?robot)', '{0}:{1}'.format(self.grasp_time-steps, self.grasp_time+steps)),
            ('(LeftGripperDownRot ?robot)', '{0}:{1}'.format(2, self.end-2)),
            ('(forall (?obj - Item)\
                (not (InGripperLeft ?robot ?item))\
            )', '0:{}'.format(0)),
            ('(forall (?obj - Item)\
                (not (InGripperLeft ?robot ?item))\
            )', '0:0'),
            #('(StationaryLeftArm ?robot)', '{0}:{0}'.format(self.grasp_time-1)),
            ('(StationaryRightArm ?robot)', '0:{}'.format(self.end-1))])
        self.eff.extend([
            ('(InGripperLeft ?robot ?item)', '{0}:{1}'.format(self.end-1, self.end)),
            ('(NearGripperLeft ?robot ?item)', '{0}:{1}'.format(self.end, self.end)),
            ('(forall (?obj - Reachable / ?item) (not (ApproachLeft ?robot ?obj)))', '{0}:{1}'.format(self.end, self.end-1)),
            ('(forall (?obj - Reachable / ?item) (not (NearApproachLeft ?robot ?obj)))', '{0}:{1}'.format(self.end, self.end-1))])


class GraspRight(Grasp):
    def __init__(self):
        super(GraspRight, self).__init__()
        steps = const.EEREACHABLE_STEPS
        self.name = 'grasp_right'
        self.pre.extend([
            ('(ApproachRight ?robot ?item)', '0:0'),
            #('(NearApproachRight ?robot ?item)', '0:0'),
            ('(not (InGripperRight ?robot ?item))', '0:0'),
            ('(not (InGripperLeft ?robot ?item))', '0:0'),
            ('(EEReachableRight ?robot ?item)', '{}:{}'.format(self.grasp_time, self.grasp_time)),
            ('(OpenGripperRight ?robot)', '{0}:{1}'.format(1,self.grasp_time-1)),
            ('(CloseGripperRight ?robot)', '{0}:{1}'.format(self.grasp_time,self.end-1)),
            ('(InGripperRight ?robot ?item)', '{0}:{1}'.format(self.grasp_time, self.grasp_time)),
            #('(RightGripperDownRot ?robot)', '{0}:{1}'.format(self.grasp_time-steps, self.grasp_time+steps)),
            ('(RightGripperDownRot ?robot)', '{0}:{1}'.format(2, self.end-2)),
            ('(forall (?obj - Item)\
                (not (InGripperRight ?robot ?item))\
            )', '0:{}'.format(self.grasp_time-1)),
            ('(forall (?obj - Item)\
                (not (InGripperRight ?robot ?item))\
            )', '0:0'),
            #('(StationaryRightArm ?robot)', '{0}:{0}'.format(self.grasp_time-1)),
            ('(StationaryLeftArm ?robot)', '0:{}'.format(self.end-1))])
        self.eff.extend([
            ('(InGripperRight ?robot ?item)', '{0}:{1}'.format(self.end-1, self.end)),
            ('(NearGripperRight ?robot ?item)', '{0}:{1}'.format(self.end, self.end)),
            ('(forall (?obj - Reachable / ?item) (not (ApproachRight ?robot ?obj)))', '{0}:{1}'.format(self.end, self.end-1)),
            ('(forall (?obj - Reachable / ?item) (not (NearApproachRight ?robot ?obj)))', '{0}:{1}'.format(self.end, self.end-1))])


class Putdown(Action):
    def __init__(self):
        self.name = 'putdown'
        self.timesteps = 5 + 2 * const.EEREACHABLE_STEPS # 2 * const.EEREACHABLE_STEPS + 11
        end = self.timesteps - 1
        self.end = end
        self.args = '(?robot - Robot ?target - Target ?item - Item ?sp - RobotPose ?ep - RobotPose)'
        putdown_time = end // 2 # const.EEREACHABLE_STEPS + 5
        approach_time = 5
        retreat_time = end-5
        self.putdown_time = putdown_time
        self.approach_time = approach_time
        self.retreat_time = retreat_time

        self.pre = [\
            ('(At ?item ?target)', '{0}:{1}'.format(putdown_time, end)),
            ('(forall (?obj - Item) (not (At ?obj ?target)))', '0:0'),
            ('(forall (?obj - Item) (not (Near ?obj ?target)))', '0:0'),
            ('(RobotAt ?robot ?sp)', '0:-1'),
            ('(not (RobotAt ?robot ?ep))', '{}:{}'.format(0, -1)),
            #('(LeftEEValid ?robot)', '{}:{}'.format(1, end-1)),
            #('(RightEEValid ?robot)', '{}:{}'.format(1, end-1)),
            ('(forall (?obj - Item) \
                (Stationary ?obj))', '{0}:{1}'.format(putdown_time, end-1)),
            ('(forall (?obj - Item)\
                (StationaryNEq ?obj ?item))', '{}:{}'.format(0, end-1)),
            ('(forall (?obs - Obstacle)\
                (StationaryW ?obs)\
            )', '{}:{}'.format(0, end-1)),
            ('(IsMP ?robot)', '0:{}'.format(end-1)),
            ('(WithinJointLimit ?robot)', '0:{}'.format(end)),
            #('(forall (?obs - Obstacle)\
            #    (not (Collides ?item ?obs))\
            #)', '0:{}'.format(end-1)),
            ('(forall (?obs - Obstacle)\
                (not (RCollides ?robot ?obs))\
            )', '1:{}'.format(self.putdown_time-1)),
            ('(forall (?obs - Obstacle)\
                (not (RCollides ?robot ?obs))\
            )', '{0}:{1}'.format(self.putdown_time+1, self.end-1)),
            ('(forall (?obj - Item)\
                (not (Obstructs ?robot ?sp ?ep ?obj))\
            )', '{}:{}'.format(putdown_time+3, end)),
            ('(forall (?obj - Item)\
                (not (ObstructsHolding ?robot ?sp ?ep ?obj ?item))\
            )', '{}:{}'.format(1, putdown_time+2))
        ]
        self.eff = [\
            ('(At ?item ?target)', '{}:{}'.format(end-1, end)) ,
            ('(Near ?item ?target)', '{}:{}'.format(end-1, end)) ,
            ('(not (RobotAt ?robot ?sp))', '{}:{}'.format(end, end-1)),
            ('(RobotAt ?robot ?ep)', '{}:{}'.format(end, end-1)),
            ('(forall (?sym1 - RobotPose)\
                (forall (?sym2 - RobotPose)\
                    (not (Obstructs ?robot ?sym1 ?sym2 ?item))\
                )\
            )', '{}:{}'.format(end, end-1)),
            ('(forall (?sym1 - Robotpose)\
                (forall (?sym2 - RobotPose)\
                    (forall (?obj - Item) (not (ObstructsHolding ?robot ?sym1 ?sym2 ?item ?obj)))\
                )\
            )', '{}:{}'.format(end, end-1))
        ]


class PutdownLeft(Putdown):
    def __init__(self):
        super(PutdownLeft, self).__init__()
        self.name = 'putdown_left'
        self.pre.extend([
            ('(ApproachLeft ?robot ?target)', '0:-1'),
            ('(NearApproachLeft ?robot ?target)', '0:0'),
            ('(not (InGripperRight ?robot ?item))', '0:0'),
            ('(InGripperLeft ?robot ?item)', '0:-1'),
            ('(NearGripperLeft ?robot ?item)', '0:0'),
            ('(InGripperLeft ?robot ?item)', '1:1'),
            ('(EEReachableLeft ?robot ?target)', '{}:{}'.format(self.putdown_time, self.putdown_time)),
            ('(CloseGripperLeft ?robot)', '{}:{}'.format(1,  self.putdown_time-1)),
            ('(OpenGripperLeft ?robot)', '{}:{}'.format(self.putdown_time,  self.end-1)),
            ('(InGripperLeft ?robot ?item)', '{}:{}'.format(self.putdown_time, self.putdown_time)),
            ('(LeftGripperDownRot ?robot)', '{0}:{1}'.format(1, self.end-1)),
            ('(forall (?obj - Item)\
                (not (InGripperLeft ?robot ?item))\
                )', '{0}:{1}'.format(self.putdown_time+1, self.end)),
            ('(StationaryRightArm ?robot)', '0:{}'.format(self.end-1))])
        self.eff.extend([
            ('(not (InGripperLeft ?robot ?item))', '{0}:{0}'.format(self.end)),
            ('(not (NearGripperLeft ?robot ?item))', '{0}:{0}'.format(self.end)),
            ])


class PutdownRight(Putdown):
    def __init__(self):
        super(PutdownRight, self).__init__()
        self.name = 'putdown_right'
        self.pre.extend([
            ('(NearApproachRight ?robot ?target)', '0:0'),
            ('(ApproachRight ?robot ?target)', '0:-1'),
            ('(not (InGripperLeft ?robot ?item))', '0:0'),
            ('(InGripperRight ?robot ?item)', '0:-1'),
            ('(InGripperRight ?robot ?item)', '1:1'),
            ('(NearGripperRight ?robot ?item)', '0:0'),
            ('(RightGripperDownRot ?robot)', '{0}:{1}'.format(1, self.end-1)),
            ('(EEReachableRight ?robot ?target)', '{}:{}'.format(self.putdown_time, self.putdown_time)),
            ('(CloseGripperRight ?robot)', '{}:{}'.format(1,  self.putdown_time-1)),
            ('(OpenGripperRight ?robot)', '{}:{}'.format(self.putdown_time,  self.end-1)),
            ('(InGripperRight ?robot ?item)', '{}:{}'.format(0, self.putdown_time)),
            ('(forall (?obj - Item)\
                (not (InGripperRight ?robot ?item))\
                )', '{0}:{1}'.format(self.putdown_time+1, self.end)),
            ('(StationaryLeftArm ?robot)', '0:{}'.format(self.end-1))])
        self.eff.extend([
            ('(not (InGripperRight ?robot ?item))', '{0}:{0}'.format(self.end)),
            ('(not (NearGripperRight ?robot ?item))', '{0}:{0}'.format(self.end)),
            ])


actions = [MoveToGraspLeft(), MoveToPutdownLeft(), GraspLeft(), PutdownLeft()]

for action in actions:
    dom_str += '\n\n'
    print(action.name)
    dom_str += action.to_str()

# removes all the extra spaces
dom_str = dom_str.replace('            ', '')
dom_str = dom_str.replace('    ', '')
dom_str = dom_str.replace('    ', '')

print(dom_str)
f = open('robot.domain', 'w')
f.write(dom_str)
