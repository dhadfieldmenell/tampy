import sys
sys.path.insert(0, '../../src/')
import core.util_classes.baxter_constants as const
dom_str = """
# AUTOGENERATED. DO NOT EDIT.
# Configuration file for CAN domain. Blank lines and lines beginning with # are filtered out.

# implicity, all types require a name
Types: Basket, BasketTarget, RobotPose, Robot, EEPose, Obstacle, Washer, WasherPose, Cloth, ClothTarget

# Define the class location of each non-standard attribute type used in the above parameter type descriptions.

Attribute Import Paths: Baxter core.util_classes.robots, Vector1d core.util_classes.matrix, Vector3d core.util_classes.matrix, ArmPose7d core.util_classes.matrix, Table core.util_classes.items, Box core.util_classes.items, Basket core.util_classes.items, Washer core.util_classes.robots, Cloth core.util_classes.items

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
pp.add('Cloth', [('geom', 'Cloth'), ('pose', 'Vector3d'), ('rotation', 'Vector3d')])
pp.add('ClothTarget', [('value', 'Vector3d'), ('rotation', 'Vector3d')])
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
pp.add('WasherPose', [('geom', 'Washer'), ('value', 'Vector3d'), ('rotation', 'Vector3d'), ('door', 'Vector1d')])
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
dp.add('BaxterClothAt', ['Cloth', 'ClothTarget'])
dp.add('BaxterRobotAt', ['Robot', 'RobotPose'])
dp.add('BaxterWasherAt', ['Washer', 'WasherPose'])
dp.add('BaxterIsMP', ['Robot'])
dp.add('BaxterWithinJointLimit', ['Robot'])
dp.add('BaxterWasherWithinJointLimit', ['Washer'])
dp.add('BaxterObjectWithinRotLimit', ['EEPose'])
dp.add('BaxterStationary', ['Basket'])
dp.add('BaxterStationaryCloth', ['Cloth'])
dp.add('BaxterStationaryWasher', ['Washer'])
dp.add('BaxterStationaryWasherDoor', ['Washer'])
dp.add('BaxterStationaryBase', ['Robot'])
dp.add('BaxterStationaryArms', ['Robot'])
dp.add('BaxterStationaryW', ['Obstacle'])
dp.add('BaxterObjRelPoseConstant', ['Basket', 'Cloth'])
dp.add('BaxterBasketGraspLeftPos', ['EEPose', 'BasketTarget'])
dp.add('BaxterBasketGraspLeftRot', ['EEPose', 'BasketTarget'])
dp.add('BaxterBasketGraspRightPos', ['EEPose', 'BasketTarget'])
dp.add('BaxterBasketGraspRightRot', ['EEPose', 'BasketTarget'])
dp.add('BaxterEEGraspValid', ['EEPose', 'Washer'])
dp.add('BaxterEEGraspValidSide', ['EEPose', 'Washer'])

dp.add('BaxterClothGraspValid', ['EEPose', 'ClothTarget'])
dp.add('BaxterCloseGripperLeft', ['Robot'])
dp.add('BaxterCloseGripperRight', ['Robot'])
dp.add('BaxterOpenGripperLeft', ['Robot'])
dp.add('BaxterOpenGripperRight', ['Robot'])
dp.add('BaxterCloseGrippers', ['Robot'])
dp.add('BaxterOpenGrippers', ['Robot'])
dp.add('BaxterObstructs', ['Robot', 'RobotPose', 'RobotPose', 'Basket'])
dp.add('BaxterObstructsHolding', ['Robot', 'RobotPose', 'RobotPose', 'Basket', 'Basket'])
dp.add('BaxterObstructsCloth', ['Robot', 'RobotPose', 'RobotPose', 'Cloth'])
dp.add('BaxterObstructsWasher', ['Robot', 'RobotPose', 'RobotPose', 'Washer'])
dp.add('BaxterObstructsHoldingCloth', ['Robot', 'RobotPose', 'RobotPose', 'Basket', 'Cloth'])
dp.add('BaxterCollides', ['Basket', 'Obstacle'])
dp.add('BaxterRCollides', ['Robot', 'Obstacle'])
dp.add('BaxterRSelfCollides', ['Robot'])
dp.add('BaxterCollidesWasher', ['Robot', 'Washer'])
dp.add('BaxterEEReachableLeftVer', ['Robot', 'RobotPose', 'EEPose'])
dp.add('BaxterEEReachableRightVer', ['Robot', 'RobotPose', 'EEPose'])
dp.add('BaxterEEApproachLeft', ['Robot', 'RobotPose', 'EEPose'])
dp.add('BaxterEEApproachRight', ['Robot', 'RobotPose', 'EEPose'])
dp.add('BaxterEERetreatLeft', ['Robot', 'RobotPose', 'EEPose'])
dp.add('BaxterEERetreatRight', ['Robot', 'RobotPose', 'EEPose'])
dp.add('BaxterBasketInGripper', ['Robot', 'Basket'])
dp.add('BaxterWasherInGripper', ['Robot', 'Washer'])
dp.add('BaxterClothInGripperLeft', ['Robot', 'Cloth'])
dp.add('BaxterClothInGripperRight', ['Robot', 'Cloth'])
dp.add('BaxterBasketLevel', ['Basket'])
dp.add('BaxterPushWasher', ['Robot', 'Washer'])
dp.add('BaxterClothTargetInWasher', ['ClothTarget', 'WasherPose'])
dp.add('BaxterClothInBasket', ['ClothTarget', 'BasketTarget'])
dp.add('BaxterPosePair', ['RobotPose', 'RobotPose'])
dp.add('BaxterClothInWasher', ['Cloth', 'Washer'])


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
            ('(BaxterRobotAt ?robot ?start)', '{}:{}'.format(0, 0)),
            ('(BaxterStationaryBase ?robot)', '{}:{}'.format(0, end-1)),
            ('(forall (?obj - Basket) (BaxterStationary ?obj))', '{}:{}'.format(0, end-1)),
            ('(forall (?obs - Cloth) (BaxterStationaryCloth ?obs))', '0:{}'.format(end-1)),
            ('(forall (?obj - Washer) (BaxterStationaryWasher ?obj))', '{}:{}'.format(0, end-1)),
            ('(forall (?obj - Washer)\
                (BaxterStationaryWasherDoor ?obj))', '{}:{}'.format(0, end-1)),
            ('(forall (?obs - Obstacle) (BaxterStationaryW ?obs))', '{}:{}'.format(0, end-1)),
            ('(forall (?basket - Basket) (BaxterBasketLevel ?basket))', '{}:{}'.format(0, end)),
            ('(BaxterIsMP ?robot)', '{}:{}'.format(0, end-1)),
            ('(BaxterWithinJointLimit ?robot)', '{}:{}'.format(0, end)),
            ('(forall (?washer - Washer) (BaxterWasherWithinJointLimit ?washer))', '0:{}'.format(end)),
            ('(forall (?obj - Basket)\
                (not (BaxterBasketInGripper ?robot ?obj))\
            )', '{}:{}'.format(0, end)),
            ('(forall (?obj - Cloth)\
                (not (BaxterClothInGripperLeft ?robot ?obj))\
            )', '{}:{}'.format(0, end)),
            ('(forall (?obj - Washer)\
                (not (BaxterWasherInGripper ?robot ?obj))\
            )', '{}:{}'.format(0, end)),
            ('(forall (?obs - Obstacle)\
                (forall (?obj - Basket)\
                    (not (BaxterCollides ?obj ?obs))\
                ))','{}:{}'.format(0, end)),
            ('(forall (?w - Obstacle) (not (BaxterRCollides ?robot ?w)))', '{}:{}'.format(0, end)),
            ('(forall (?obj - Basket)\
                (not (BaxterObstructs ?robot ?start ?end ?obj)))', '{}:{}'.format(0, end-1)),
            ('(forall (?obj - Washer)\
                (not (BaxterObstructsWasher ?robot ?start ?end ?obj)))', '{}:{}'.format(0, end-1))
        ]
        self.eff = [\
            (' (not (BaxterRobotAt ?robot ?start))', '{}:{}'.format(end, end-1)),
            ('(BaxterRobotAt ?robot ?end)', '{}:{}'.format(end, end))]

class ClothGrasp(Action):
    def __init__(self):
        self.name = 'cloth_grasp'
        self.timesteps = 2 * const.EEREACHABLE_STEPS + 11
        end = self.timesteps - 1
        self.args = '(?robot - Robot ?cloth - Cloth ?target - ClothTarget ?sp - RobotPose ?ee_left - EEPose ?ep - RobotPose)'
        grasp_time = const.EEREACHABLE_STEPS+5
        approach_time = 5
        retreat_time = end-5
        self.pre = [\
            ('(BaxterClothAt ?cloth ?target)', '0:0'),
            ('(BaxterRobotAt ?robot ?sp)', '0:0'),
            ('(BaxterPosePair ?sp ?ep)', '0:0'),
            ('(forall (?obj - Basket)\
                (not (BaxterBasketInGripper ?robot ?obj))\
            )', '{}:{}'.format(0, end)),
            ('(forall (?obj - Cloth)\
                (not (BaxterClothInGripperLeft ?robot ?obj))\
            )', '{}:{}'.format(0, grasp_time-1)),
            ('(forall (?obj - Washer)\
                (not (BaxterWasherInGripper ?robot ?obj))\
            )', '{}:{}'.format(0, end)),

            ('(BaxterEEReachableLeftVer ?robot ?sp ?ee_left)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(BaxterClothGraspValid ?ee_left ?target)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(BaxterOpenGripperLeft ?robot)', '{}:{}'.format(0,  grasp_time-1)),
            ('(BaxterStationaryCloth ?cloth)', '{}:{}'.format(0, grasp_time-1)),
            ('(forall (?obj - Basket) \
                (BaxterStationary ?obj)\
            )', '0:{}'.format(end-1)),
            ('(forall (?obs - Washer)\
                (BaxterStationaryWasher ?obs)\
            )', '{}:{}'.format(0, end-1)),
            ('(forall (?obs - Washer) (BaxterStationaryWasherDoor ?obs))', '0:{}'.format(end-1)),
            ('(forall (?obs - Obstacle)\
                (BaxterStationaryW ?obs)\
            )', '{}:{}'.format(0, end-1)),
            ('(BaxterStationaryBase ?robot)', '{}:{}'.format(0, end-1)),
            ('(BaxterIsMP ?robot)', '0:{}'.format(end-1)),
            ('(BaxterWithinJointLimit ?robot)', '0:{}'.format(end)),
            ('(forall (?washer - Washer) (BaxterWasherWithinJointLimit ?washer))', '0:{}'.format(end)),
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
            )', '0:{}'.format(end))
        ]
        self.eff = [\
            ('(not (BaxterOpenGripperLeft ?robot))', '{}:{}'.format(end,  end-1)),
            ('(BaxterCloseGripperLeft ?robot)', '{}:{}'.format(grasp_time,  end)),
            ('(BaxterClothInGripperLeft ?robot ?cloth)', '{}:{}'.format(grasp_time, end)),
            ('(not (BaxterClothAt ?cloth ?target))', '{}:{}'.format(end, end-1)) ,
            ('(not (BaxterStationaryCloth ?cloth))', '{}:{}'.format(grasp_time, end-1)),
            ('(not (BaxterRobotAt ?robot ?sp))', '{}:{}'.format(end, end-1)),
            ('(BaxterRobotAt ?robot ?ep)', '{}:{}'.format(end, end)),
            ('(forall (?sym1 - Robotpose)\
                (forall (?sym2 - RobotPose)\
                    (forall (?obj - Basket) (not (BaxterObstructsHoldingCloth ?robot ?sym1 ?sym2 ?obj ?cloth)))\
                )\
            )', '{}:{}'.format(end, end-1))
        ]

class MoveHoldingCloth(Action):
    def __init__(self):
        self.name = 'moveholding_cloth'
        self.timesteps = 20
        end = self.timesteps - 1
        self.args = '(?robot - Robot ?start - RobotPose ?end - RobotPose ?cloth - Cloth)'
        self.pre = [\
            ('(BaxterRobotAt ?robot ?start)', '0:0'),
            ('(BaxterCloseGripperLeft ?robot)', '{}:{}'.format(0,  end)),
            ('(BaxterClothInGripperLeft ?robot ?cloth)', '{}:{}'.format(end, end)),
            ('(forall (?obj - Basket)\
                (not (BaxterObstructsHoldingCloth ?robot ?start ?end ?obj ?cloth))\
            )', '0:{}'.format(end)),
            ('(forall (?obj - Washer)\
                (not (BaxterObstructsWasher ?robot ?start ?end ?obj)))', '{}:{}'.format(0, end-1)),
            ('(forall (?obj - Basket)\
                (BaxterStationary ?obj))', '{}:{}'.format(0, end-1)),
            ('(not (BaxterStationaryCloth ?cloth))', '{}:{}'.format(0, end-1)),
            ('(BaxterStationaryBase ?robot)', '{}:{}'.format(0, end-1)),
            ('(forall (?obs - Washer) (BaxterStationaryWasher ?obs))', '0:{}'.format(end-1)),
            ('(forall (?obs - Washer) (BaxterStationaryWasherDoor ?obs))', '0:{}'.format(end-1)),
            ('(forall (?obs - Obstacle) (BaxterStationaryW ?obs))', '0:{}'.format(end-1)),
            ('(BaxterIsMP ?robot)', '0:{}'.format(end-1)),
            ('(BaxterWithinJointLimit ?robot)', '0:{}'.format(end)),
            ('(forall (?washer - Washer) (BaxterWasherWithinJointLimit ?washer))', '0:{}'.format(end)),
            ('(forall (?obs - Obstacle)\
                (forall (?obj - Basket)\
                    (not (BaxterCollides ?obj ?obs))\
                )\
            )', '0:{}'.format(end)),
            ('(forall (?obs - Obstacle) (not (BaxterRCollides ?robot ?obs)))', '0:{}'.format(end))
        ]
        self.eff = [\
            ('(not (BaxterRobotAt ?robot ?start))', '{}:{}'.format(end, end-1)),
            ('(BaxterRobotAt ?robot ?end)', '{}:{}'.format(end, end))
        ]

class ClothPutdown(Action):
    def __init__(self):
        self.name = 'cloth_putdown'
        self.timesteps = 2 * const.EEREACHABLE_STEPS + 11
        end = self.timesteps - 1
        self.args = '(?robot - Robot ?cloth - Cloth ?target - ClothTarget ?sp - RobotPose ?ee_left - EEPose ?ep - RobotPose)'
        putdown_time = const.EEREACHABLE_STEPS+5
        approach_time = 5
        retreat_time = end-5
        self.pre = [\
            ('(BaxterRobotAt ?robot ?sp)', '0:0'),
            ('(BaxterPosePair ?sp ?ep)', '0:0'),
            ('(BaxterEEReachableLeftVer ?robot ?sp ?ee_left)', '{}:{}'.format(putdown_time, putdown_time)),
            ('(BaxterClothInGripperLeft ?robot ?cloth)', '{}:{}'.format(0, putdown_time)),
            ('(BaxterClothGraspValid ?ee_left ?target)', '{}:{}'.format(putdown_time, putdown_time)),
            ('(BaxterCloseGripperLeft ?robot)', '{}:{}'.format(0,  putdown_time-1)),
            ('(forall (?obj - Basket) \
                (BaxterStationary ?obj)\
            )', '0:{}'.format(end-1)),
            ('(not (BaxterStationaryCloth ?cloth))', '{}:{}'.format(0, putdown_time-1)),
            ('(forall (?obs - Washer)\
                (BaxterStationaryWasher ?obs)\
            )', '{}:{}'.format(0, end-1)),
            ('(forall (?obs - Washer) (BaxterStationaryWasherDoor ?obs))', '0:{}'.format(end-1)),
            ('(forall (?obs - Obstacle)\
                (BaxterStationaryW ?obs)\
            )', '{}:{}'.format(0, end-1)),
            ('(BaxterStationaryBase ?robot)', '{}:{}'.format(0, end-1)),
            ('(BaxterIsMP ?robot)', '0:{}'.format(end-1)),
            ('(BaxterWithinJointLimit ?robot)', '0:{}'.format(end)),
            ('(forall (?washer - Washer) (BaxterWasherWithinJointLimit ?washer))', '0:{}'.format(end)),
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
            )', '{}:{}'.format(0, end)),
            ('(forall (?sym1 - RobotPose)\
                (forall (?sym2 - RobotPose)\
                    (not (BaxterObstructsCloth ?robot ?sym1 ?sym2 ?cloth))\
                )\
            )', '{}:{}'.format(end, end-1))
        ]
        self.eff = [\
            ('(BaxterClothAt ?cloth ?target)', '{}:{}'.format(end, end-1)) ,
            ('(BaxterStationaryCloth ?cloth)', '{}:{}'.format(putdown_time, end-1)),
            ('(not (BaxterRobotAt ?robot ?sp))', '{}:{}'.format(end, end-1)),
            ('(BaxterRobotAt ?robot ?ep)', '{}:{}'.format(end, end)),
            ('(not (BaxterCloseGripperLeft ?robot))', '{}:{}'.format(end,  end-1)),
            ('(BaxterOpenGripperLeft ?robot)', '{}:{}'.format(putdown_time,  end)),
            ('(not (BaxterClothInGripperLeft ?robot ?cloth))', '{}:{}'.format(end, end))
        ]

class BasketGrasp(Action):
    def __init__(self):
        self.name = 'basket_grasp'
        self.timesteps = 2 * const.EEREACHABLE_STEPS + 11
        end = self.timesteps - 1
        self.args = '(?robot - Robot ?basket - Basket ?target - BasketTarget ?sp - RobotPose ?ee_left - EEPose ?ee_right - EEPose ?ep - RobotPose)'
        grasp_time = const.EEREACHABLE_STEPS + 5
        approach_time = 5
        retreat_time = end-5
        self.pre = [\
            ('(BaxterAt ?basket ?target)', '0:{}'.format(grasp_time)),
            ('(BaxterPosePair ?sp ?ep)', '0:0'),
            ('(BaxterRobotAt ?robot ?sp)', '0:0'),
            ('(BaxterEEReachableLeftVer ?robot ?sp ?ee_left)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(BaxterEEReachableRightVer ?robot ?sp ?ee_right)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(BaxterOpenGripperLeft ?robot)', '{}:{}'.format(0,  grasp_time-1)),
            ('(BaxterOpenGripperRight ?robot)', '{}:{}'.format(0,  grasp_time-1)),
            ('(BaxterBasketGraspLeftPos ?ee_left ?target)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(BaxterBasketGraspLeftRot ?ee_left ?target)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(BaxterBasketGraspRightPos ?ee_right ?target)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(BaxterBasketGraspRightRot ?ee_right ?target)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(forall (?obj - Basket)\
                (not (BaxterBasketInGripper ?robot ?obj))\
            )', '0:{}'.format(grasp_time-1)),
            ('(forall (?obj - Cloth)\
                (not (BaxterClothInGripperLeft ?robot ?obj))\
            )', '{}:{}'.format(0, end)),
            ('(forall (?obj - Washer)\
                (not (BaxterWasherInGripper ?robot ?obj))\
            )', '{}:{}'.format(0, end)),
            ('(BaxterBasketLevel ?basket)', '{}:{}'.format(0, end)),
            ('(BaxterStationary ?basket)', '0:{}'.format(grasp_time-1)),
            ('(forall (?obs - Washer) (BaxterStationaryWasher ?obs))', '{}:{}'.format(0, end-1)),
            ('(forall (?obs - Washer) (BaxterStationaryWasherDoor ?obs))', '0:{}'.format(end-1)),
            ('(forall (?obs - Obstacle)\
                (BaxterStationaryW ?obs)\
            )', '{}:{}'.format(0, end-1)),
            ('(BaxterStationaryBase ?robot)', '{}:{}'.format(0, end-1)),
            # ('(BaxterStationaryBase ?robot)', '{}:{}'.format(approach_time, retreat_time-1)),
            ('(BaxterIsMP ?robot)', '0:{}'.format(end-1)),
            ('(BaxterWithinJointLimit ?robot)', '0:{}'.format(end)),
            ('(forall (?washer - Washer) (BaxterWasherWithinJointLimit ?washer))', '0:{}'.format(end)),
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
            ('(not (BaxterStationary ?basket))', '{}:{}'.format(grasp_time, end-1)),
            ('(not (BaxterAt ?basket ?target))', '{}:{}'.format(end, end-1)) ,
            ('(not (BaxterRobotAt ?robot ?sp))', '{}:{}'.format(end, end-1)),
            ('(BaxterRobotAt ?robot ?ep)', '{}:{}'.format(end, end)),
            ('(BaxterBasketInGripper ?robot ?basket)', '{}:{}'.format(grasp_time, end)),
            ('(not (BaxterOpenGripperLeft ?robot))', '{}:{}'.format(end,  end-1)),
            ('(not (BaxterOpenGripperRight ?robot))', '{}:{}'.format(end,  end-1)),
            ('(BaxterCloseGripperLeft ?robot)', '{}:{}'.format(grasp_time,  end)),
            ('(BaxterCloseGripperRight ?robot)', '{}:{}'.format(grasp_time,  end)),

            ('(forall (?cloth - Cloth) (BaxterObjRelPoseConstant ?basket ?cloth))', '0:{}'.format(end-1)),
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

class MoveHoldingBasket(Action):
    def __init__(self):
        self.name = 'moveholding_basket'
        self.timesteps = 20
        end = self.timesteps - 1
        self.args = '(?robot - Robot ?start - RobotPose ?end - RobotPose ?basket - Basket)'
        self.pre = [\
            ('(BaxterRobotAt ?robot ?start)', '0:0'),
            ('(BaxterCloseGripperLeft ?robot)', '{}:{}'.format(0,  end)),
            ('(BaxterCloseGripperRight ?robot)', '{}:{}'.format(0,  end)),
            ('(BaxterBasketInGripper ?robot ?basket)', '0:{}'.format(end)),
            ('(forall (?obj - Basket)\
                (not (BaxterObstructsHolding ?robot ?start ?end ?obj ?basket))\
            )', '0:{}'.format(end)),
            ('(forall (?obj - Washer)\
                (not (BaxterObstructsWasher ?robot ?start ?end ?obj)))', '{}:{}'.format(0, end-1)),
            ('(BaxterStationaryBase ?robot)', '{}:{}'.format(0, end-1)),
            ('(not (BaxterStationary ?basket))', '{}:{}'.format(0, end-1)),
            ('(forall (?obs - Washer) (BaxterStationaryWasher ?obs))', '0:{}'.format(end-1)),
            ('(forall (?obs - Washer) (BaxterStationaryWasherDoor ?obs))', '0:{}'.format(end-1)),
            ('(forall (?obs - Obstacle) (BaxterStationaryW ?obs))', '0:{}'.format(end-1)),
            ('(BaxterIsMP ?robot)', '0:{}'.format(end-1)),
            ('(BaxterWithinJointLimit ?robot)', '0:{}'.format(end)),
            ('(forall (?washer - Washer) (BaxterWasherWithinJointLimit ?washer))', '0:{}'.format(end)),
            ('(forall (?cloth - Cloth) (BaxterObjRelPoseConstant ?basket ?cloth))', '0:{}'.format(end-1)),
            ('(BaxterBasketLevel ?basket)', '{}:{}'.format(0, end)),
            ('(forall (?obs - Obstacle)\
                (forall (?obj - Basket)\
                    (not (BaxterCollides ?obj ?obs))\
                )\
            )', '0:{}'.format(end)),
            ('(forall (?obs - Obstacle) (not (BaxterRCollides ?robot ?obs)))', '0:{}'.format(end))
        ]
        self.eff = [\
            ('(not (BaxterRobotAt ?robot ?start))', '{}:{}'.format(end, end-1)),
            ('(BaxterRobotAt ?robot ?end)', '{}:{}'.format(end, end))
        ]

class BasketPutdown(Action):
    def __init__(self):
        self.name = 'basket_putdown'
        self.timesteps = 2 * const.EEREACHABLE_STEPS + 11
        end = self.timesteps - 1
        self.args = '(?robot - Robot ?basket - Basket ?target - BasketTarget ?sp - RobotPose ?ee_left - EEPose ?ee_right - EEPose ?ep - RobotPose)'
        putdown_time = const.EEREACHABLE_STEPS + 5
        approach_time = 5
        retreat_time = end - 5
        self.pre = [\
            ('(BaxterRobotAt ?robot ?sp)', '0:0'),
            ('(BaxterPosePair ?sp ?ep)', '0:0'),
            ('(BaxterEEReachableLeftVer ?robot ?sp ?ee_left)', '{}:{}'.format(putdown_time, putdown_time)),
            ('(BaxterEEReachableRightVer ?robot ?sp ?ee_right)', '{}:{}'.format(putdown_time, putdown_time)),
            ('(BaxterCloseGripperLeft ?robot)', '{}:{}'.format(0,  putdown_time)),
            ('(BaxterCloseGripperRight ?robot)', '{}:{}'.format(0,  putdown_time)),
            ('(BaxterBasketGraspLeftPos ?ee_left ?target)', '{}:{}'.format(putdown_time, putdown_time)),
            ('(BaxterBasketGraspLeftRot ?ee_left ?target)', '{}:{}'.format(putdown_time, putdown_time)),
            ('(BaxterBasketGraspRightPos ?ee_right ?target)', '{}:{}'.format(putdown_time, putdown_time)),
            ('(BaxterBasketGraspRightRot ?ee_right ?target)', '{}:{}'.format(putdown_time, putdown_time)),
            ('(BaxterBasketInGripper ?robot ?basket)', '{}:{}'.format(0, putdown_time)),
            ('(BaxterBasketLevel ?basket)', '{}:{}'.format(0, end)),
            ('(not (BaxterStationary ?basket))', '{}:{}'.format(0, putdown_time-1)),
            ('(forall (?obs - Obstacle)\
                (BaxterStationaryW ?obs)\
            )', '{}:{}'.format(0, end-1)),
            ('(forall (?obs - Washer)\
                (BaxterStationaryWasher ?obs)\
            )', '{}:{}'.format(0, end-1)),
            ('(forall (?obs - Washer) (BaxterStationaryWasherDoor ?obs))', '0:{}'.format(end-1)),
            ('(BaxterStationaryBase ?robot)', '{}:{}'.format(0, end-1)),
            # ('(BaxterStationaryBase ?robot)', '{}:{}'.format(approach_time, retreat_time-1)),
            ('(BaxterIsMP ?robot)', '0:{}'.format(end-1)),
            ('(BaxterWithinJointLimit ?robot)', '0:{}'.format(end)),
            ('(forall (?washer - Washer) (BaxterWasherWithinJointLimit ?washer))', '0:{}'.format(end)),
            ('(forall (?cloth - Cloth) (BaxterObjRelPoseConstant ?basket ?cloth))', '0:{}'.format(end-1)),
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
            ('(BaxterStationary ?basket)', '{}:{}'.format(putdown_time, end-1)),
            ('(BaxterAt ?basket ?target)', '{}:{}'.format(putdown_time, end)),
            ('(not (BaxterRobotAt ?robot ?sp))', '{}:{}'.format(end, end-1)),
            ('(BaxterRobotAt ?robot ?ep)', '{}:{}'.format(end, end)),
            ('(not (BaxterBasketInGripper ?robot ?basket))', '{}:{}'.format(end, end-1)),
            ('(not (BaxterCloseGripperLeft ?robot))', '{}:{}'.format(end,  end-1)),
            ('(not (BaxterCloseGripperRight ?robot))', '{}:{}'.format(end,  end-1)),
            ('(BaxterOpenGripperLeft ?robot)', '{}:{}'.format(putdown_time+1,  end)),
            ('(BaxterOpenGripperRight ?robot)', '{}:{}'.format(putdown_time+1,  end)),
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

class OpenDoor(Action):
    def __init__(self):
        self.name = 'open_door'
        self.timesteps = 2*(const.EEREACHABLE_STEPS+6) + 25
        end = self.timesteps - 1
        self.args = '(?robot - Robot ?washer - Washer ?sp - RobotPose ?ee_approach - EEPose ?ee_retreat - EEPose ?ep - RobotPose ?wsp - WasherPose ?wep - WasherPose)'
        grasp_time = const.EEREACHABLE_STEPS + 5
        retreat_time = end - const.EEREACHABLE_STEPS - 5
        self.pre = [\
            ('(BaxterRobotAt ?robot ?sp)', '0:0'),
            ('(BaxterPosePair ?sp ?ep)', '0:0'),
            ('(BaxterWasherAt ?washer ?wsp)', '0:{}'.format(grasp_time)),
            ('(BaxterEEApproachLeft ?robot ?sp ?ee_approach)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(BaxterEEGraspValid ?ee_approach ?washer)', '{}:{}'.format(grasp_time, grasp_time)),
            ('(BaxterEEGraspValid ?ee_retreat ?washer)', '{}:{}'.format(retreat_time, retreat_time)),
            ('(BaxterOpenGripperLeft ?robot)', '{}:{}'.format(0,  grasp_time-1)),
            ('(BaxterOpenGripperLeft ?robot)', '{}:{}'.format(retreat_time+1,  end)),
            ('(BaxterEERetreatLeft ?robot ?ep ?ee_retreat)', '{}:{}'.format(retreat_time, retreat_time)),

            ('(forall (?obj - Basket)\
                (not (BaxterBasketInGripper ?robot ?obj))\
            )', '0:{}'.format(end)),
            ('(forall (?obj - Cloth)\
                (not (BaxterClothInGripperLeft ?robot ?obj))\
            )', '{}:{}'.format(0, end)),
            ('(forall (?obj - Washer)\
                (not (BaxterWasherInGripper ?robot ?obj))\
            )', '{}:{}'.format(0, grasp_time-1)),

            ('(BaxterStationaryWasher ?washer)', '0:{}'.format(end-1)),
            ('(BaxterStationaryWasherDoor ?washer)', '0:{}'.format(grasp_time-1)),
            ('(forall (?obj - Basket) \
                (BaxterStationary ?obj)\
            )', '0:{}'.format(end-1)),
            ('(forall (?obj - Cloth) \
                (BaxterStationaryCloth ?obj)\
            )', '0:{}'.format(end-1)),
            ('(forall (?obs - Obstacle)\
                (BaxterStationaryW ?obs)\
            )', '{}:{}'.format(0, end-1)),
            # ('(BaxterStationaryBase ?robot)', '{}:{}'.format(0, grasp_time-1)),
            ('(BaxterStationaryBase ?robot)', '{}:{}'.format(0, end-1)),
            ('(BaxterIsMP ?robot)', '0:{}'.format(end-1)),
            # ('(BaxterWasherIsMP ?washer)', '0:{}'.format(end-1)),
            ('(BaxterWithinJointLimit ?robot)', '0:{}'.format(end)),
            ('(BaxterWasherWithinJointLimit ?washer)', '0:{}'.format(end)),
            ('(not (BaxterObstructsWasher ?robot ?sp ?ep ?washer))', '0:{}'.format(end)),
            ('(forall (?obs - Obstacle)\
                (forall (?obj - Basket)\
                    (not (BaxterCollides ?obj ?obs))\
                )\
            )', '0:{}'.format(end-1)),
            ('(forall (?obs - Obstacle)\
                (not (BaxterRCollides ?robot ?obs))\
            )', '0:{}'.format(end)),
            ('(not (BaxterRSelfCollides ?robot))', '0:{}'.format(end)),
            ('(forall (?obj - Basket)\
                (not (BaxterObstructs ?robot ?sp ?ep ?obj))\
            )', '0:{}'.format(grasp_time-1))
        ]
        self.eff = [\
            ('(not (BaxterRobotAt ?robot ?sp))', '{}:{}'.format(end, end-1)),
            ('(not (BaxterWasherAt ?washer ?wsp))', '{}:{}'.format(end, end-1)),
            ('(BaxterRobotAt ?robot ?ep)', '{}:{}'.format(end, end)),
            ('(BaxterWasherAt ?washer ?wep)', '{}:{}'.format(retreat_time, end)),
            ('(BaxterCloseGripperLeft ?robot)', '{}:{}'.format(grasp_time,  retreat_time)),
            ('(not (BaxterCloseGripperLeft ?robot))', '{}:{}'.format(end,  end-1)),
            ('(BaxterWasherInGripper ?robot ?washer)', '{}:{}'.format(grasp_time, retreat_time)),
            ('(not (BaxterWasherInGripper ?robot ?washer))', '{}:{}'.format(end, end-1)),
            ('(not (BaxterStationaryWasherDoor ?washer))', '0:{}'.format(grasp_time, retreat_time-1)),
            ('(BaxterStationaryWasherDoor ?washer)', '{}:{}'.format(retreat_time-1, end-1)),
            ('(forall (?sym1 - RobotPose)\
                (forall (?sym2 - RobotPose)\
                (forall (?obj - Basket)\
                    (not (BaxterObstructs ?robot ?sym1 ?sym2 ?obj)))\
                )\
            )', '{}:{}'.format(end, end-1))
        ]

class PutIntoWasher(Action):
    def __init__(self):
        self.name = 'put_into_washer'
        self.timesteps = 30
        end = self.timesteps - 1
        self.args = '(?robot - Robot ?washer - Washer ?wp - WasherPose ?cloth - Cloth ?target - ClothTarget ?sp - RobotPose ?ee_left - EEPose ?ep - RobotPose)'
        putdown_time = 10
        approach_time = 5
        retreat_time = end-5
        self.pre = [\
            ('(BaxterPosePair ?sp ?ep)', '0:0'),
            ('(BaxterRobotAt ?robot ?sp)', '0:0'),
            ('(BaxterWasherAt ?washer ?wp)', '0:0'),
            ('(BaxterClothInGripperLeft ?robot ?cloth)', '{}:{}'.format(0, putdown_time)),
            ('(BaxterClothGraspValid ?ee_left ?target)', '{}:{}'.format(putdown_time, putdown_time)),
            ('(BaxterCloseGripperLeft ?robot)', '{}:{}'.format(0,  putdown_time-1)),
            ('(BaxterClothTargetInWasher ?target ?wp)', '{}:{}'.format(0, 0)),
            ('(not (BaxterStationaryCloth ?cloth))', '{}:{}'.format(0, putdown_time-1)),
            ('(forall (?obj - Basket) \
                (BaxterStationary ?obj)\
            )', '0:{}'.format(end-1)),
            ('(forall (?obs - Washer)\
                (BaxterStationaryWasher ?obs)\
            )', '{}:{}'.format(0, end-1)),
            ('(forall (?obs - Washer) (BaxterStationaryWasherDoor ?obs))', '0:{}'.format(end-1)),
            ('(forall (?obs - Obstacle)\
                (BaxterStationaryW ?obs)\
            )', '{}:{}'.format(0, end-1)),
            ('(BaxterStationaryBase ?robot)', '{}:{}'.format(0, end-1)),
            ('(BaxterIsMP ?robot)', '0:{}'.format(end-1)),
            ('(BaxterWithinJointLimit ?robot)', '0:{}'.format(end)),
            ('(BaxterWasherWithinJointLimit ?washer)', '0:{}'.format(end)),
            ('(forall (?obs - Obstacle)\
                (forall (?obj - Basket)\
                    (not (BaxterCollides ?obj ?obs))\
                )\
            )', '0:{}'.format(end-1)),
            ('(not (BaxterRSelfCollides ?robot))', '0:{}'.format(end)),
            ('(forall (?obs - Obstacle)\
                (not (BaxterRCollides ?robot ?obs))\
            )', '0:{}'.format(end)),
            ('(forall (?obj - Basket)\
                (not (BaxterObstructs ?robot ?sp ?ep ?obj))\
            )', '{}:{}'.format(0, end)),
            ('(not (BaxterCollidesWasher ?robot ?washer))', '{}:{}'.format(0, end))
        ]
        self.eff = [\
            ('(not (BaxterCloseGripperLeft ?robot))', '{}:{}'.format(end,  end-1)),
            ('(BaxterOpenGripperLeft ?robot)', '{}:{}'.format(putdown_time,  end)),
            ('(BaxterClothInWasher ?cloth ?washer)', '{}:{}'.format(end, end)) ,
            ('(BaxterClothAt ?cloth ?target)', '{}:{}'.format(retreat_time, end)) ,
            ('(not (BaxterRobotAt ?robot ?sp))', '{}:{}'.format(end, end-1)),
            ('(BaxterRobotAt ?robot ?ep)', '{}:{}'.format(end, end)),
            ('(not (BaxterClothInGripperLeft ?robot ?cloth))', '{}:{}'.format(end, end)),
            ('(BaxterStationaryCloth ?cloth)', '{}:{}'.format(putdown_time, end-1)),
            ('(forall (?sym1 - RobotPose)\
                (forall (?sym2 - RobotPose)\
                    (not (BaxterObstructsCloth ?robot ?sym1 ?sym2 ?cloth))\
                )\
            )', '{}:{}'.format(end, end-1))
        ]

actions = [Move(), ClothGrasp(), MoveHoldingCloth(), ClothPutdown(), BasketGrasp(), MoveHoldingBasket(), BasketPutdown(), OpenDoor(), PutIntoWasher()]
for action in actions:
    dom_str += '\n\n'
    dom_str += action.to_str()

# removes all the extra spaces
dom_str = dom_str.replace('            ', '')
dom_str = dom_str.replace('    ', '')
dom_str = dom_str.replace('    ', '')

# print dom_str
f = open('laundry_hl.domain', 'w')
f.write(dom_str)

print "Domain File Generated Successfully "
