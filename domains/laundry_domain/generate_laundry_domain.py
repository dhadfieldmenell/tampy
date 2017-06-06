import sys
sys.path.insert(0, '../../src/')
import core.util_classes.baxter_constants as const
dom_str = """
# AUTOGENERATED. DO NOT EDIT.
# Configuration file for CAN domain. Blank lines and lines beginning with # are filtered out.

# implicity, all types require a name
Types: Basket, BasketTarget, RobotPose, Robot, EEPose, Obstacle, Washer

# Define the class location of each non-standard attribute type used in the above parameter type descriptions.

Attribute Import Paths: Baxter core.util_classes.robots, Vector1d core.util_classes.matrix, Vector3d core.util_classes.matrix, ArmPose7d core.util_classes.matrix, Table core.util_classes.items, Box core.util_classes.items, Basket core.util_classes.items, Washer core.util_classes.robots

Predicates Import Path: core.util_classes.baxter_predicates

"""

class PrimitivePredicates(object):
    def __init__(self):
        self.attr_dict = {}

    def add(self, name, attrs):
        self.attr_dict[name] = attrs

    def get_str(self):
        prim_str = 'Primitive Predicates: '
        first = True
        for name, attrs in self.attr_dict.iteritems():
            for attr_name, attr_type in attrs:
                pred_str = attr_name + ', ' + name + ', ' + attr_type
                if first:
                    prim_str += pred_str
                    first = False
                else:
                    prim_str += '; ' + pred_str
        return prim_str

pp = PrimitivePredicates()
pp.add('Basket', [('geom', 'Basket'), ('pose', 'Vector3d'), ('rotation', 'Vector3d')])
pp.add('BasketTarget', [('geom', 'Basket'), ('value', 'Vector3d'), ('rotation', 'Vector3d')])
pp.add('RobotPose', [('value', 'Vector1d'),
                    ('lArmPose', 'ArmPose7d'),
                    ('lGripper', 'Vector1d'),
                    ('rArmPose', 'ArmPose7d'),
                    ('rGripper', 'Vector1d')])
pp.add('Robot', [('geom', 'Baxter'),
                ('pose', 'Vector1d'),
                ('lArmPose', 'ArmPose7d'),
                ('lGripper', 'Vector1d'),
                ('rArmPose', 'ArmPose7d'),
                ('rGripper', 'Vector1d')])
pp.add('EEPose', [('value', 'Vector3d'), ('rotation', 'Vector3d')])
pp.add('Washer', [('geom', 'Washer'), ('pose', 'Vector3d'), ('rotation', 'Vector3d'), ('door', 'Vector1d')])
pp.add('Obstacle', [('geom', 'Box'), ('pose', 'Vector3d'), ('rotation', 'Vector3d')])
dom_str += pp.get_str() + '\n\n'

class DerivatedPredicates(object):
    def __init__(self):
        self.pred_dict = {}

    def add(self, name, args):
        self.pred_dict[name] = args

    def get_str(self):
        prim_str = 'Derived Predicates: '

        first = True
        for name, args in self.pred_dict.iteritems():
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
dp.add('BaxterAt', ['Basket', 'BasketTarget'])
dp.add('BaxterRobotAt', ['Robot', 'RobotPose'])
dp.add('BaxterEEReachablePos', ['Robot', 'RobotPose', 'EEPose'])
dp.add('BaxterEEReachableRot', ['Robot', 'RobotPose', 'EEPose'])
dp.add('BaxterEEReachableVerLeftPos', ['Robot', 'RobotPose', 'EEPose'])
dp.add('BaxterEEReachableVerLeftRot', ['Robot', 'RobotPose', 'EEPose'])
dp.add('BaxterEEReachableVerRightPos', ['Robot', 'RobotPose', 'EEPose'])
dp.add('BaxterEEReachableVerRightRot', ['Robot', 'RobotPose', 'EEPose'])
dp.add('BaxterBasketInGripperPos', ['Robot', 'Basket'])
dp.add('BaxterBasketInGripperRot', ['Robot', 'Basket'])
dp.add('BaxterBasketInContact', ['Robot', 'EEPose', 'EEPose', 'BasketTarget'])
dp.add('BaxterBasketOpenGripper', ['Robot', 'EEPose', 'EEPose', 'BasketTarget'])
dp.add('BaxterObstructs', ['Robot', 'RobotPose', 'RobotPose', 'Basket'])
dp.add('BaxterObstructsHolding', ['Robot', 'RobotPose', 'RobotPose', 'Basket', 'Basket'])
dp.add('BaxterBasketGraspLeftPos', ['EEPose', 'BasketTarget'])
dp.add('BaxterBasketGraspLeftRot', ['EEPose', 'BasketTarget'])
dp.add('BaxterBasketGraspRightPos', ['EEPose', 'BasketTarget'])
dp.add('BaxterBasketGraspRightRot', ['EEPose', 'BasketTarget'])
dp.add('BaxterBasketLevel', ['Basket'])
dp.add('BaxterStationary', ['Basket'])
dp.add('BaxterStationaryW', ['Obstacle'])
dp.add('BaxterStationaryArms', ['Robot'])
dp.add('BaxterStationaryBase', ['Robot'])
dp.add('BaxterIsMP', ['Robot'])
dp.add('BaxterWithinJointLimit', ['Robot'])
dp.add('BaxterCollides', ['Basket', 'Obstacle'])
dp.add('BaxterRCollides', ['Robot', 'Obstacle'])

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
        self.timesteps = 20
        end = self.timesteps - 1
        self.args = '(?robot - Robot ?start - RobotPose ?end - RobotPose)'
        self.pre = [\
            ('(forall (?obj - Basket)\
                (not (BaxterBasketInGripperPos ?robot ?obj))\
            )', '{}:{}'.format(0, 0)),
            ('(BaxterRobotAt ?robot ?start)', '{}:{}'.format(0, 0)),
            ('(forall (?obj - Basket )\
                (not (BaxterObstructs ?robot ?start ?end ?obj)))', '{}:{}'.format(0, end-1)),
            ('(forall (?obj - Basket)\
                (BaxterStationary ?obj))', '{}:{}'.format(0, end-1)),
            ('(forall (?obs - Obstacle) (BaxterStationaryW ?obs))', '{}:{}'.format(0, end-1)),
            ('(BaxterBasketOpenGripper ?robot ?ee_left ?ee_right ?target)', '{}:{}'.format(0, end)),
            ('(BaxterIsMP ?robot)', '{}:{}'.format(0, end-1)),
            ('(BaxterWithinJointLimit ?robot)', '{}:{}'.format(0, end)),
            ('(forall (?obs - Obstacle)\
                (forall (?obj - Basket)\
                    (not (BaxterCollides ?obj ?obs))\
                ))','{}:{}'.format(0, end)),
            ('(forall (?w - Obstacle) (not (BaxterRCollides ?robot ?w)))', '{}:{}'.format(0, end))
        ]
        self.eff = [\
            (' (not (BaxterRobotAt ?robot ?start))', '{}:{}'.format(end, end-1)),
            ('(BaxterRobotAt ?robot ?end)', '{}:{}'.format(end, end))]

class MoveHolding(Action):
    def __init__(self):
        self.name = 'movetoholding'
        self.timesteps = 20
        end = self.timesteps - 1
        self.args = '(?robot - Robot ?start - RobotPose ?end - RobotPose ?basket - Basket)'
        self.pre = [\
            ('(BaxterRobotAt ?robot ?start)', '0:0'),
            ('(BaxterBasketInGripperPos ?robot ?basket)', '0:{}'.format(end)),
            ('(BaxterBasketInGripperRot ?robot ?basket)', '0:{}'.format(end)),
            ('(forall (?obj - Basket)\
                (not (BaxterObstructsHolding ?robot ?start ?end ?obj ?basket))\
            )', '0:{}'.format(end)),
            ('(forall (?obs - Obstacle) (BaxterStationaryW ?obs))', '0:{}'.format(end-1)),
            ('(BaxterBasketInContact ?robot ?ee_left ?ee_right ?target)', '{}:{}'.format(0, end)),
            ('(BaxterIsMP ?robot)', '0:{}'.format(end-1)),
            ('(BaxterWithinJointLimit ?robot)', '0:{}'.format(end)),
            ('(forall (?obs - Obstacle)\
                (forall (?obj - Basket)\
                    (not (BaxterCollides ?obj ?obs))\
                )\
            )', '0:{}'.format(end)),
            ('(forall (?obs - Obstacle) (not (BaxterRCollides ?robot ?obs)))', '0:{}'.format(end))
        ]
        self.eff = [\
            (' (BaxterBasketInGripperPos ?robot ?basket)', '0:{}'.format(end)),
            ('(BaxterBasketInGripperRot ?robot ?basket)', '0:{}'.format(end)),
            ('(not (BaxterRobotAt ?robot ?start))', '{}:{}'.format(end, end-1)),
            ('(BaxterRobotAt ?robot ?end)', '{}:{}'.format(end, end))
        ]

class Grasp(Action):
    def __init__(self):
        self.name = 'basket_grasp'
        self.timesteps = 2 * const.EEREACHABLE_STEPS + 1
        end = self.timesteps - 1
        self.args = '(?robot - Robot ?basket - Basket ?target - BasketTarget ?sp - RobotPose ?ee_left - EEPose ?ee_right - EEPose ?ep - RobotPose)'
        grasp_time = const.EEREACHABLE_STEPS
        approach_time = 0
        retreat_time = end
        self.pre = [\
            ('(BaxterAt ?basket ?target)', '0:0'),
            ('(BaxterRobotAt ?robot ?sp)', '0:0'),
            ('(BaxterEEReachableVerLeftPos ?robot ?sp ?ee_left)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(BaxterEEReachableVerLeftRot ?robot ?sp ?ee_left)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(BaxterEEReachableVerRightPos ?robot ?sp ?ee_right)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(BaxterEEReachableVerRightRot ?robot ?sp ?ee_right)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(BaxterBasketOpenGripper ?robot ?ee_left ?ee_right ?target)', '{}:{}'.format(0,  grasp_time-1)),
            ('(BaxterBasketInContact ?robot ?ee_left ?ee_right ?target)', '{}:{}'.format(grasp_time,  end)),
            ('(BaxterBasketGraspLeftPos ?ee_left ?target)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(BaxterBasketGraspLeftRot ?ee_left ?target)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(BaxterBasketGraspRightPos ?ee_right ?target)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(BaxterBasketGraspRightRot ?ee_right ?target)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(BaxterBasketInGripperPos ?robot ?basket)', '{}:{}'.format(grasp_time, end)),
            ('(BaxterBasketInGripperRot ?robot ?basket)', '{}:{}'.format(grasp_time, end)),
            ('(forall (?obj - Basket)\
                (not (BaxterBasketInGripperPos ?robot ?basket))\
            )', '0:{}'.format(grasp_time-1)),
            ('(forall (?obj - Basket) \
                (BaxterStationary ?obj)\
            )', '0:{}'.format(grasp_time-1)),
            ('(forall (?obs - Obstacle)\
                (BaxterStationaryW ?obs)\
            )', '{}:{}'.format(0, end-1)),
            ('(BaxterStationaryBase ?robot)', '{}:{}'.format(approach_time, retreat_time-1)),
            ('(BaxterIsMP ?robot)', '0:{}'.format(end-1)),
            ('(BaxterWithinJointLimit ?robot)', '0:{}'.format(end)),
            ('(forall (?obs - Obstacle)\
                (forall (?obj - Basket)\
                    (not (BaxterCollides ?obj ?obs))\
                )\
            )', '0:{}'.format(end-1)),
            ('(forall (?obs - Obstacle)\
                (not (BaxterRCollides ?robot ?obs))\
            )', '0:{}'.format(end)),

            ('(forall (?obj - Basket)\
                (not (BaxterObstructs ?robot ?sp ?ep ?obj))\
            )', '0:{}'.format(grasp_time-1)),
            ('(forall (?obj - Basket)\
                (not (BaxterObstructsHolding ?robot ?sp ?ep ?obj ?basket))\
            )', '{}:{}'.format(grasp_time, end))
        ]
        self.eff = [\
            (' (not (BaxterAt ?basket ?target))', '{}:{}'.format(end, end-1)) ,
            ('(not (BaxterRobotAt ?robot ?sp))', '{}:{}'.format(end, end-1)),
            ('(BaxterRobotAt ?robot ?ep)', '{}:{}'.format(end, end)),
            ('(BaxterBasketInGripperPos ?robot ?basket)', '{}:{}'.format(end, end)),
            ('(BaxterBasketInGripperRot ?robot ?basket)', '{}:{}'.format(end, end)),

            ('(forall (?sym1 - RobotPose)\
                (forall (?sym2 - RobotPose)\
                    (not (BaxterObstructs ?robot ?sym1 ?sym2 ?basket))\
                )\
            )', '{}:{}'.format(end, end-1)),
            ('(forall (?sym1 - Robotpose)\
                (forall (?sym2 - RobotPose)\
                    (forall (?obj - Basket) (not (BaxterObstructsHolding ?robot ?sym1 ?sym2 ?basket ?obj)))\
                )\
            )', '{}:{}'.format(end, end-1))
        ]

class Putdown(Action):
    def __init__(self):
        self.name = 'basket_putdown'
        self.timesteps = 2 * const.EEREACHABLE_STEPS + 1
        end = self.timesteps - 1
        self.args = '(?robot - Robot ?basket - Basket ?target - BasketTarget ?sp - RobotPose ?ee_left - EEPose ?ee_right - EEPose ?ep - RobotPose)'
        putdown_time = const.EEREACHABLE_STEPS
        approach_time = 0
        retreat_time = end
        self.pre = [\
            ('(BaxterRobotAt ?robot ?sp)', '0:0'),
            ('(BaxterEEReachableVerLeftPos ?robot ?sp ?ee_left)', '{}:{}'.format(putdown_time, putdown_time)),
            ('(BaxterEEReachableVerLeftRot ?robot ?sp ?ee_left)', '{}:{}'.format(putdown_time, putdown_time)),
            ('(BaxterEEReachableVerRightPos ?robot ?sp ?ee_right)', '{}:{}'.format(putdown_time, putdown_time)),
            ('(BaxterEEReachableVerRightRot ?robot ?sp ?ee_right)', '{}:{}'.format(putdown_time, putdown_time)),
            ('(BaxterBasketOpenGripper ?robot ?ee_left ?ee_right ?target)', '{}:{}'.format(0,  putdown_time)),
            ('(BaxterBasketInContact ?robot ?ee_left ?ee_right ?target)', '{}:{}'.format(putdown_time+1,  end)),
            ('(BaxterBasketGraspLeftPos ?ee_left ?target)', '{}:{}'.format(putdown_time, putdown_time)),
            ('(BaxterBasketGraspLeftRot ?ee_left ?target)', '{}:{}'.format(putdown_time, putdown_time)),
            ('(BaxterBasketGraspRightPos ?ee_right ?target)', '{}:{}'.format(putdown_time, putdown_time)),
            ('(BaxterBasketGraspRightRot ?ee_right ?target)', '{}:{}'.format(putdown_time, putdown_time)),
            ('(BaxterBasketInGripperPos ?robot ?basket)', '{}:{}'.format(0, putdown_time)),
            ('(BaxterBasketInGripperRot ?robot ?basket)', '{}:{}'.format(0, putdown_time)),
            ('(forall (?obj - Basket)\
                (not (BaxterBasketInGripperPos ?robot ?basket))\
            )', '{}:{}'.format(putdown_time+1, end)),
            ('(forall (?obj - Basket) \
                (BaxterStationary ?obj)\
            )', '{}:{}'.format(putdown_time+1, end)),
            ('(forall (?obs - Obstacle)\
                (BaxterStationaryW ?obs)\
            )', '{}:{}'.format(0, end-1)),
            ('(BaxterStationaryBase ?robot)', '{}:{}'.format(approach_time, retreat_time-1)),
            ('(BaxterIsMP ?robot)', '0:{}'.format(end-1)),
            ('(BaxterWithinJointLimit ?robot)', '0:{}'.format(end)),
            ('(forall (?obs - Obstacle)\
                (forall (?obj - Basket)\
                    (not (BaxterCollides ?obj ?obs))\
                )\
            )', '0:{}'.format(end-1)),
            ('(forall (?obs - Obstacle)\
                (not (BaxterRCollides ?robot ?obs))\
            )', '0:{}'.format(end)),

            ('(forall (?obj - Basket)\
                (not (BaxterObstructs ?robot ?sp ?ep ?obj))\
            )', '{}:{}'.format(putdown_time+1, end)),
            ('(forall (?obj - Basket)\
                (not (BaxterObstructsHolding ?robot ?sp ?ep ?obj ?basket))\
            )', '{}:{}'.format(0, putdown_time))
        ]
        self.eff = [\
            (' (BaxterAt ?basket ?target)', '{}:{}'.format(end, end)),
            ('(not (BaxterRobotAt ?robot ?sp))', '{}:{}'.format(end, end-1)),
            ('(BaxterRobotAt ?robot ?ep)', '{}:{}'.format(end, end)),
            ('(not \
                (BaxterBasketInGripperPos ?robot ?basket))', '{}:{}'.format(end, end-1)),
            ('(not \
                (BaxterBasketInGripperRot ?robot ?basket))', '{}:{}'.format(end, end-1)),

            ('(forall (?sym1 - RobotPose)\
                (forall (?sym2 - RobotPose)\
                    (not (BaxterObstructs ?robot ?sym1 ?sym2 ?basket))\
                )\
            )', '{}:{}'.format(end, end-1)),
            ('(forall (?sym1 - Robotpose)\
                (forall (?sym2 - RobotPose)\
                    (forall (?obj - Basket) (not (BaxterObstructsHolding ?robot ?sym1 ?sym2 ?basket ?obj)))\
                )\
            )', '{}:{}'.format(end, end-1))
        ]

actions = [Move(), MoveHolding(), Grasp(), Putdown()]
for action in actions:
    dom_str += '\n\n'
    dom_str += action.to_str()

# removes all the extra spaces
dom_str = dom_str.replace('            ', '')
dom_str = dom_str.replace('    ', '')
dom_str = dom_str.replace('    ', '')

print dom_str
f = open('laundry.domain', 'w')
f.write(dom_str)