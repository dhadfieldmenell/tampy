import sys
sys.path.insert(0, '../../src/')
from core.util_classes.pr2_predicates import EEREACHABLE_STEPS

dom_str = """
# AUTOGENERATED. DO NOT EDIT.
# Configuration file for CAN domain. Blank lines and lines beginning with # are filtered out.

# implicity, all types require a name
Types: Can, Target, RobotPose, Robot, EEPose, Obstacle

# Define the class location of each non-standard attribute type used in the above parameter type descriptions.
Attribute Import Paths: RedCan core.util_classes.items, BlueCan core.util_classes.items, PR2 core.util_classes.robots, Vector1d core.util_classes.matrix, Vector3d core.util_classes.matrix, PR2ArmPose core.util_classes.matrix, Table core.util_classes.items, Box core.util_classes.items

Predicates Import Path: core.util_classes.pr2_predicates

"""

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
pp.add('Can', [('geom', 'RedCan'), ('pose', 'Vector3d'), ('rotation', 'Vector3d')])
pp.add('Target', [('geom', 'BlueCan'), ('value', 'Vector3d'), ('rotation', 'Vector3d')])
pp.add('RobotPose', [('value', 'Vector3d'),
                    ('backHeight', 'Vector1d'),
                    ('lArmPose', 'PR2ArmPose'),
                    ('lGripper', 'Vector1d'),
                    ('rArmPose', 'PR2ArmPose'),
                    ('rGripper', 'Vector1d')])
pp.add('Robot', [('geom', 'PR2'),
                ('pose', 'Vector3d'),
                ('backHeight', 'Vector1d'),
                ('lArmPose', 'PR2ArmPose'),
                ('lGripper', 'Vector1d'),
                ('rArmPose', 'PR2ArmPose'),
                ('rGripper', 'Vector1d')])
pp.add('EEPose', [('value', 'Vector3d'), ('rotation', 'Vector3d')])
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
dp.add('PR2At', ['Can', 'Target'])
dp.add('PR2RobotAt', ['Robot', 'RobotPose'])
dp.add('PR2EEReachablePosRight', ['Robot', 'RobotPose', 'EEPose'])
dp.add('PR2EEReachableRotRight', ['Robot', 'RobotPose', 'EEPose'])
dp.add('PR2EEReachablePosLeft', ['Robot', 'RobotPose', 'EEPose'])
dp.add('PR2EEReachableRotLeft', ['Robot', 'RobotPose', 'EEPose'])
dp.add('PR2InGripperPosRight', ['Robot', 'Can'])
dp.add('PR2InGripperRotRight', ['Robot', 'Can'])
dp.add('PR2InGripperPosLeft', ['Robot', 'Can'])
dp.add('PR2InGripperRotLeft', ['Robot', 'Can'])
dp.add('PR2InContactRight', ['Robot', 'EEPose', 'Target'])
dp.add('PR2InContactLeft', ['Robot', 'EEPose', 'Target'])
dp.add('PR2Obstructs', ['Robot', 'RobotPose', 'RobotPose', 'Can'])
dp.add('PR2ObstructsHolding', ['Robot', 'RobotPose', 'RobotPose', 'Can', 'Can'])
dp.add('PR2GraspValidPos', ['EEPose', 'Target'])
dp.add('PR2GraspValidRot', ['EEPose', 'Target'])
dp.add('PR2Stationary', ['Can'])
dp.add('PR2StationaryW', ['Obstacle'])
dp.add('PR2StationaryNEq', ['Can', 'Can'])
dp.add('PR2StationaryArms', ['Robot'])
dp.add('PR2StationaryBase', ['Robot'])
dp.add('PR2IsMP', ['Robot'])
dp.add('PR2WithinJointLimit', ['Robot'])
dp.add('PR2Collides', ['Can', 'Obstacle'])
dp.add('PR2RCollides', ['Robot', 'Obstacle'])
dp.add('PR2BothEndsInGripper', ['Robot', 'Can'])

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
        self.timesteps = 30
        end = self.timesteps - 1
        self.args = '(?robot - Robot ?start - RobotPose ?end - RobotPose)'
        self.pre = [\
            ('(forall (?c - Can)\
                (not (PR2InGripperPosRight ?robot ?c))\
            )', '0:0'),
            ('(forall (?c - Can)\
                (not (PR2InGripperRotRight ?robot ?c))\
            )', '0:0'),
            ('(forall (?c - Can)\
                (not (PR2InGripperPosLeft ?robot ?c))\
            )', '0:0'),
            ('(forall (?c - Can)\
                (not (PR2InGripperRotLeft ?robot ?c))\
            )', '0:0'),
            ('(PR2RobotAt ?robot ?start)', '0:0'),
            ('(forall (?obj - Can )\
                (not (PR2Obstructs ?robot ?start ?end ?obj)))', '0:{}'.format(end-1)),
            ('(forall (?obj - Can)\
                (PR2Stationary ?obj))', '0:{}'.format(end-1)),
            ('(forall (?w - Obstacle) (PR2StationaryW ?w))', '0:{}'.format(end-1)),
            # ('(PR2StationaryArms ?robot)', '0:{}'.format(end-1)),
            ('(PR2StationaryBase ?robot)', '0:{}'.format(end-1)),
            ('(PR2IsMP ?robot)', '0:{}'.format(end-1)),
            ('(PR2WithinJointLimit ?robot)', '0:{}'.format(end)),
            # ('(forall (?w     - Obstacle)\
            #     (forall (?obj - Can)\
            #         (not (PR2Collides ?obj ?w))\
            #     ))','0:19')
            ('(forall (?w - Obstacle) (not (PR2RCollides ?robot ?w)))', '0:{}'.format(end))
        ]
        self.eff = [\
            ('(not (PR2RobotAt ?robot ?start))', '{}:{}'.format(end, end)),
            ('(PR2RobotAt ?robot ?end)', '{}:{}'.format(end, end))]

class MoveHoldingRight(Action):
    def __init__(self):
        self.name = 'movetoholding_right'
        self.timesteps = 20
        self.args = '(?robot - Robot ?start - RobotPose ?end - RobotPose ?c - Can)'
        self.pre = [\
            ('(PR2RobotAt ?robot ?start)', '0:0'),
            ('(PR2InGripperPosRight ?robot ?c)', '0:19'),
            ('(PR2InGripperRotRight ?robot ?c)', '0:19'),
            # ('(forall (?obj - Can)\
            #     (not (PR2ObstructsHolding ?robot ?start ?end ?obj ?c))\
            # )', '0:19'),
            ('(forall (?obj - Can) (PR2StationaryNEq ?obj ?c))', '0:18'),
            ('(forall (?w - Obstacle) (PR2StationaryW ?w))', '0:18'),
            ('(PR2StationaryArms ?robot)', '0:18'),
            ('(PR2IsMP ?robot)', '0:18'),
            ('(PR2WithinJointLimit ?robot)', '0:19')
            # ('(forall (?w - Obstacle)\
            #     (forall (?obj - Can)\
            #         (not (PR2Collides ?obj ?w))\
            #     )\
            # )', '0:19')
            # ('(forall (?w - Obstacle) (not (PR2RCollides ?robot ?w)))', '0:19')
        ]
        self.eff = [\
            ('(not (PR2RobotAt ?robot ?start))', '19:19'),
            ('(PR2RobotAt ?robot ?end)', '19:19')
        ]

class MoveHoldingLeft(Action):
    def __init__(self):
        self.name = 'movetoholding_left'
        self.timesteps = 20
        self.args = '(?robot - Robot ?start - RobotPose ?end - RobotPose ?c - Can)'
        self.pre = [\
            ('(PR2RobotAt ?robot ?start)', '0:0'),
            ('(PR2InGripperPosLeft ?robot ?c)', '0:19'),
            ('(PR2InGripperRotLeft ?robot ?c)', '0:19'),
            # ('(forall (?obj - Can)\
            #     (not (PR2ObstructsHolding ?robot ?start ?end ?obj ?c))\
            # )', '0:19'),
            ('(forall (?obj - Can) (PR2StationaryNEq ?obj ?c))', '0:18'),
            ('(forall (?w - Obstacle) (PR2StationaryW ?w))', '0:18'),
            ('(PR2StationaryArms ?robot)', '0:18'),
            ('(PR2IsMP ?robot)', '0:18'),
            ('(PR2WithinJointLimit ?robot)', '0:19')
            # ('(forall (?w - Obstacle)\
            #     (forall (?obj - Can)\
            #         (not (PR2Collides ?obj ?w))\
            #     )\
            # )', '0:19')
            # ('(forall (?w - Obstacle) (not (PR2RCollides ?robot ?w)))', '0:19')
        ]
        self.eff = [\
            ('(not (PR2RobotAt ?robot ?start))', '19:19'),
            ('(PR2RobotAt ?robot ?end)', '19:19')
        ]

class MoveBothHoldingCloth(Action):
    def __init__(self):
        self.name = 'move_both_holding_cloth'
        self.timesteps = 20
        end = self.timesteps - 1
        self.args = '(?robot - Robot  ?cloth - Can ?target - Target ?start - RobotPose ?end - RobotPose)'
        self.pre = [\
            ('(PR2RobotAt ?robot ?start)', '0:0'),
            ('(PR2BothEndsInGripper ?robot ?cloth)', '0:{}'.format(end)),
            ('(PR2CloseGrippers ?robot)', '0:{}'.format(end)),
            ('(forall (?obj - Basket)\
                (not (PR2Obstructs ?robot ?start ?end ?obj))\
            )', '0:{}'.format(end)),
            ('(forall (?obj - Basket)\
                (PR2Stationary ?obj))', '{}:{}'.format(0, end-1)),
            ('(PR2StationaryBase ?robot)', '{}:{}'.format(0, end-1)),
            ('(forall (?obs - Obstacle) (PR2StationaryW ?obs))', '0:{}'.format(end-1)),
            ('(PR2IsMP ?robot)', '0:{}'.format(end-1)),
            ('(PR2WithinJointLimit ?robot)', '0:{}'.format(end)),
            ('(forall (?obs - Obstacle)\
                (forall (?obj - Basket)\
                    (not (PR2Collides ?obj ?obs))\
                )\
            )', '0:{}'.format(end)),
            # ('(not (PR2RSelfCollides ?robot))', '1:{}'.format(end-1)),
            ('(forall (?obs - Obstacle) (not (PR2RCollides ?robot ?obs)))', '0:{}'.format(end))
        ]
        self.eff = [\
            ('(not (PR2RobotAt ?robot ?start))', '{}:{}'.format(end, end-1)),
            ('(PR2RobotAt ?robot ?end)', '{}:{}'.format(end, end)),
            ('(PR2At ?cloth ?target)', '{}:{}'.format(end, end))
        ]

class GraspRight(Action):
    def __init__(self):
        self.name = 'grasp_right'
        self.timesteps = 40
        self.args = '(?robot - Robot ?can - Can ?target - Target ?sp - RobotPose ?ee - EEPose ?ep - RobotPose)'
        steps = EEREACHABLE_STEPS
        grasp_time = self.timesteps/2
        approach_time = grasp_time-steps
        retreat_time = grasp_time+steps
        self.pre = [\
            ('(PR2At ?can ?target)', '0:0'),
            ('(PR2RobotAt ?robot ?sp)', '0:0'),
            ('(PR2EEReachablePosRight ?robot ?sp ?ee)', '{}:{}'.format(grasp_time, grasp_time)),
            # ('(PR2EEReachableRotRight ?robot ?sp ?ee)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(PR2EEReachableRotRight ?robot ?sp ?ee)', '{}:{}'.format(approach_time, retreat_time)),
            # ('(PR2EEReachableRot ?robot ?sp ?ee)', '16:24'),
            # TODO: not sure why InContact to 39 fails
            ('(PR2InContactRight ?robot ?ee ?target)', '{}:38'.format(grasp_time)),
            ('(PR2GraspValidPos ?ee ?target)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(PR2GraspValidRot ?ee ?target)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(forall (?obj - Can)\
                (not (PR2InGripperPosRight ?robot ?obj))\
            )', '0:0'),
            ('(forall (?obj - Can)\
                (not (PR2InGripperRotRight ?robot ?obj))\
            )', '0:0'),
            ('(forall (?obj - Can) \
                (PR2Stationary ?obj)\
            )', '0:{}'.format(grasp_time-1)),
            ('(forall (?obj - Can) (PR2StationaryNEq ?obj ?can))', '{}:38'.format(grasp_time)),
            ('(forall (?w - Obstacle)\
                (PR2StationaryW ?w)\
            )', '0:38'),
            # ('(PR2StationaryBase ?robot)', '17:22'),
            ('(PR2StationaryBase ?robot)', '{}:{}'.format(approach_time, retreat_time-1)),
            # ('(PR2StationaryBase ?robot)', '0:38'),
            # ('(PR2IsMP ?robot)', '0:38'),
            ('(PR2WithinJointLimit ?robot)', '0:39'),
            # ('(forall (?w - Obstacle)\
            #     (forall (?obj - Can)\
            #         (not (PR2Collides ?obj ?w))\
            #     )\
            # )', '0:38'),
            ('(forall (?w - Obstacle)\
                (not (PR2RCollides ?robot ?w))\
            )', '0:39'),
            ('(forall (?obj - Can)\
                (not (PR2Obstructs ?robot ?sp ?ep ?obj))\
            )', '0:{}'.format(approach_time)),
            ('(forall (?obj - Can)\
                (not (PR2ObstructsHolding ?robot ?sp ?ep ?obj ?can))\
            )', '{}:39'.format(approach_time+1))
        ]
        self.eff = [\
            ('(not (PR2At ?can ?target))', '39:38') ,
            ('(not (PR2RobotAt ?robot ?sp))', '39:38'),
            ('(PR2RobotAt ?robot ?ep)', '39:39'),
            ('(PR2InGripperPosRight ?robot ?can)', '{}:39'.format(grasp_time+1)),
            ('(PR2InGripperRotRight ?robot ?can)', '{}:39'.format(grasp_time+1)),
            ('(forall (?sym1 - RobotPose)\
                (forall (?sym2 - RobotPose)\
                    (not (PR2Obstructs ?robot ?sym1 ?sym2 ?can))\
                )\
            )', '39:38'),
            ('(forall (?sym1 - Robotpose)\
                (forall (?sym2 - RobotPose)\
                    (forall (?obj - Can) (not (PR2ObstructsHolding ?robot ?sym1 ?sym2 ?can ?obj)))\
                )\
            )', '39:38')
        ]

class GraspLeft(Action):
    def __init__(self):
        self.name = 'grasp_left'
        self.timesteps = 40
        self.args = '(?robot - Robot ?can - Can ?target - Target ?sp - RobotPose ?ee - EEPose ?ep - RobotPose)'
        steps = EEREACHABLE_STEPS
        grasp_time = self.timesteps/2
        approach_time = grasp_time-steps
        retreat_time = grasp_time+steps
        self.pre = [\
            ('(PR2At ?can ?target)', '0:0'),
            ('(PR2RobotAt ?robot ?sp)', '0:0'),
            ('(PR2EEReachablePosLeft ?robot ?sp ?ee)', '{}:{}'.format(grasp_time, grasp_time)),
            # ('(PR2EEReachableRotLeft ?robot ?sp ?ee)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(PR2EEReachableRotLeft ?robot ?sp ?ee)', '{}:{}'.format(approach_time, retreat_time)),
            # ('(PR2EEReachableRot ?robot ?sp ?ee)', '16:24'),
            # TODO: not sure why InContact to 39 fails
            ('(PR2InContactLeft ?robot ?ee ?target)', '{}:38'.format(grasp_time)),
            ('(PR2GraspValidPos ?ee ?target)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(PR2GraspValidRot ?ee ?target)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(forall (?obj - Can)\
                (not (PR2InGripperPosLeft ?robot ?obj))\
            )', '0:0'),
            ('(forall (?obj - Can)\
                (not (PR2InGripperRotLeft ?robot ?obj))\
            )', '0:0'),
            ('(forall (?obj - Can) \
                (PR2Stationary ?obj)\
            )', '0:{}'.format(grasp_time-1)),
            ('(forall (?obj - Can) (PR2StationaryNEq ?obj ?can))', '{}:38'.format(grasp_time)),
            ('(forall (?w - Obstacle)\
                (PR2StationaryW ?w)\
            )', '0:38'),
            # ('(PR2StationaryBase ?robot)', '17:22'),
            ('(PR2StationaryBase ?robot)', '{}:{}'.format(approach_time, retreat_time-1)),
            # ('(PR2StationaryBase ?robot)', '0:38'),
            # ('(PR2IsMP ?robot)', '0:38'),
            ('(PR2WithinJointLimit ?robot)', '0:39'),
            # ('(forall (?w - Obstacle)\
            #     (forall (?obj - Can)\
            #         (not (PR2Collides ?obj ?w))\
            #     )\
            # )', '0:38'),
            ('(forall (?w - Obstacle)\
                (not (PR2RCollides ?robot ?w))\
            )', '0:39'),
            ('(forall (?obj - Can)\
                (not (PR2Obstructs ?robot ?sp ?ep ?obj))\
            )', '0:{}'.format(approach_time)),
            ('(forall (?obj - Can)\
                (not (PR2ObstructsHolding ?robot ?sp ?ep ?obj ?can))\
            )', '{}:39'.format(approach_time+1))
        ]
        self.eff = [\
            ('(not (PR2At ?can ?target))', '39:38') ,
            ('(not (PR2RobotAt ?robot ?sp))', '39:38'),
            ('(PR2RobotAt ?robot ?ep)', '39:39'),
            ('(PR2InGripperPosLeft ?robot ?can)', '{}:39'.format(grasp_time+1)),
            ('(PR2InGripperRotLeft ?robot ?can)', '{}:39'.format(grasp_time+1)),
            ('(forall (?sym1 - RobotPose)\
                (forall (?sym2 - RobotPose)\
                    (not (PR2Obstructs ?robot ?sym1 ?sym2 ?can))\
                )\
            )', '39:38'),
            ('(forall (?sym1 - Robotpose)\
                (forall (?sym2 - RobotPose)\
                    (forall (?obj - Can) (not (PR2ObstructsHolding ?robot ?sym1 ?sym2 ?can ?obj)))\
                )\
            )', '39:38')
        ]

class PutdownRight(Action):
    def __init__(self):
        self.name = 'putdown_right'
        self.timesteps = 20
        self.args = '(?robot - Robot ?can - Can ?target - Target ?sp - RobotPose ?ee - EEPose ?ep - RobotPose)'
        self.pre = [\
            ('(PR2RobotAt ?robot ?sp)', '0:0'),
            ('(PR2EEReachablePosRight ?robot ?sp ?ee)', '10:10'),
            ('(PR2EEReachableRotRight ?robot ?sp ?ee)', '10:10'),
            ('(PR2InContactRight ?robot ?ee ?target)', '0:10'),
            ('(PR2GraspValidPos ?ee ?target)', '0:0'),
            ('(PR2GraspValidRot ?ee ?target)', '0:0'),
            ('(PR2InGripperPosRight ?robot ?can)', '0:10'),
            ('(PR2InGripperRotRight ?robot ?can)', '0:10'),
            # is the line below so that we have to use a new ee with the target?
            # ('(not (PR2InContact ?robot ?ee ?target))', '0:0'),
            ('(forall (?obj - Can)\
                (PR2Stationary ?obj)\
            )', '10:18'),
            ('(forall (?obj - Can) (PR2StationaryNEq ?obj ?can))', '0:9'),
            ('(forall (?w - Obstacle)\
                (PR2StationaryW ?w)\
            )', '0:18'),
            ('(PR2StationaryBase ?robot)', '0:18'),
            ('(PR2IsMP ?robot)', '0:18'),
            ('(PR2WithinJointLimit ?robot)', '0:19')
            # ('(forall (?w - Obstacle)\
            #     (forall (?obj - Can)\
            #         (not (PR2Collides ?obj ?w))\
            #     )\
            # )', '0:18'),
            # ('(forall (?w - Obstacle)\
            #     (not (PR2RCollides ?robot ?w))\
            # )', '0:19'),
            # ('(forall (?obj - Can)\
            #     (not (PR2ObstructsHolding ?robot ?sp ?ep ?obj ?can))\
            # )', '0:19'),
            # ('(forall (?obj - Can)\
            #     (not (PR2Obstructs ?robot ?sp ?ep ?obj))\
            # )', '19:19')
        ]
        self.eff = [\
            ('(not (PR2RobotAt ?robot ?sp))', '19:19'),
            ('(PR2RobotAt ?robot ?ep)', '19:19'),
            ('(PR2At ?can ?target)', '19:19'),
            ('(not (PR2InGripperPosRight ?robot ?can))', '19:19'),
            ('(not (PR2InGripperRotRight ?robot ?can))', '19:19')
        ]

class PutdownLeft(Action):
    def __init__(self):
        self.name = 'putdown_left'
        self.timesteps = 20
        self.args = '(?robot - Robot ?can - Can ?target - Target ?sp - RobotPose ?ee - EEPose ?ep - RobotPose)'
        self.pre = [\
            ('(PR2RobotAt ?robot ?sp)', '0:0'),
            ('(PR2EEReachablePosLeft ?robot ?sp ?ee)', '10:10'),
            ('(PR2EEReachableRotLeft ?robot ?sp ?ee)', '10:10'),
            ('(PR2InContactLeft ?robot ?ee ?target)', '0:10'),
            ('(PR2GraspValidPos ?ee ?target)', '0:0'),
            ('(PR2GraspValidRot ?ee ?target)', '0:0'),
            ('(PR2InGripperPosLeft ?robot ?can)', '0:10'),
            ('(PR2InGripperRotLeft ?robot ?can)', '0:10'),
            # is the line below so that we have to use a new ee with the target?
            # ('(not (PR2InContact ?robot ?ee ?target))', '0:0'),
            ('(forall (?obj - Can)\
                (PR2Stationary ?obj)\
            )', '10:18'),
            ('(forall (?obj - Can) (PR2StationaryNEq ?obj ?can))', '0:9'),
            ('(forall (?w - Obstacle)\
                (PR2StationaryW ?w)\
            )', '0:18'),
            ('(PR2StationaryBase ?robot)', '0:18'),
            ('(PR2IsMP ?robot)', '0:18'),
            ('(PR2WithinJointLimit ?robot)', '0:19')
            # ('(forall (?w - Obstacle)\
            #     (forall (?obj - Can)\
            #         (not (PR2Collides ?obj ?w))\
            #     )\
            # )', '0:18'),
            # ('(forall (?w - Obstacle)\
            #     (not (PR2RCollides ?robot ?w))\
            # )', '0:19'),
            # ('(forall (?obj - Can)\
            #     (not (PR2ObstructsHolding ?robot ?sp ?ep ?obj ?can))\
            # )', '0:19'),
            # ('(forall (?obj - Can)\
            #     (not (PR2Obstructs ?robot ?sp ?ep ?obj))\
            # )', '19:19')
        ]
        self.eff = [\
            ('(not (PR2RobotAt ?robot ?sp))', '19:19'),
            ('(PR2RobotAt ?robot ?ep)', '19:19'),
            ('(PR2At ?can ?target)', '19:19'),
            ('(not (PR2InGripperPosLeft ?robot ?can))', '19:19'),
            ('(not (PR2InGripperRotLeft ?robot ?can))', '19:19')
        ]

actions = [Move(), MoveHoldingRight(), MoveHoldingLeft(), MoveBothHoldingCloth(), GraspRight(), GraspLEft(), PutdownRight(), PutDownLeft()]
for action in actions:
    dom_str += '\n\n'
    dom_str += action.to_str()

# removes all the extra spaces
dom_str = dom_str.replace('            ', '')
dom_str = dom_str.replace('    ', '')
dom_str = dom_str.replace('    ', '')

print(dom_str)
f = open('can.domain', 'w')
f.write(dom_str)
