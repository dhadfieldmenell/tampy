dom_str = """
# AUTOGENERATED. DO NOT EDIT.
# Configuration file for CAN domain. Blank lines and lines beginning with # are filtered out.

# implicity, all types require a name
Types: Can, Target, RobotPose, Robot, EEPose, Obstacle

# Define the class location of each non-standard attribute type used in the above parameter type descriptions.

Attribute Import Paths: RedCan core.util_classes.can, BlueCan core.util_classes.can, Baxter core.util_classes.robots, Vector1d core.util_classes.matrix, Vector3d core.util_classes.matrix, ArmPose7d core.util_classes.matrix, Table core.util_classes.table, Box core.util_classes.box

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
        for name, attrs in self.attr_dict.items():
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
        for name, args in self.pred_dict.items():
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
dp.add('BaxterAt', ['Can', 'Target'])
dp.add('BaxterRobotAt', ['Robot', 'RobotPose'])
dp.add('BaxterEEReachablePos', ['Robot', 'RobotPose', 'EEPose'])
dp.add('BaxterEEReachableRot', ['Robot', 'RobotPose', 'EEPose'])
dp.add('BaxterInGripperPos', ['Robot', 'Can'])
dp.add('BaxterInGripperRot', ['Robot', 'Can'])
dp.add('BaxterInContact', ['Robot', 'EEPose', 'Target'])
dp.add('BaxterObstructs', ['Robot', 'RobotPose', 'RobotPose', 'Can'])
dp.add('BaxterObstructsHolding', ['Robot', 'RobotPose', 'RobotPose', 'Can', 'Can'])
dp.add('BaxterGraspValidPos', ['EEPose', 'Target'])
dp.add('BaxterGraspValidRot', ['EEPose', 'Target'])
dp.add('BaxterStationary', ['Can'])
dp.add('BaxterStationaryW', ['Obstacle'])
dp.add('BaxterStationaryNEq', ['Can', 'Can'])
dp.add('BaxterStationaryArms', ['Robot'])
dp.add('BaxterStationaryBase', ['Robot'])
dp.add('BaxterIsMP', ['Robot'])
dp.add('BaxterWithinJointLimit', ['Robot'])
dp.add('BaxterCollides', ['Can', 'Obstacle'])
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

class Grasp(Action):
    def __init__(self):
        self.name = 'grasp'
        self.timesteps = 40
        self.args = '(?robot - Robot ?can - Can ?target - Target ?sp - RobotPose ?ee - EEPose ?ep - RobotPose)'
        self.pre = [\
            ('(BaxterAt ?can ?target)', '0:0'),
            ('(BaxterRobotAt ?robot ?sp)', '0:0'),
            ('(BaxterEEReachablePos ?robot ?sp ?ee)', '20:20'),
            ('(BaxterEEReachableRot ?robot ?sp ?ee)', '20:20'),
            ('(BaxterEEReachableRot ?robot ?sp ?ee)', '16:24'),
            # TODO: not sure why BaxterInContact to 39 fails
            ('(BaxterInContact ?robot ?ee ?target)', '20:38'),
            ('(BaxterGraspValidPos ?ee ?target)', '20:20'),
            ('(BaxterGraspValidRot ?ee ?target)', '20:20'),
            ('(forall (?obj - Can)\
                (not (BaxterInGripperPos ?robot ?obj))\
            )', '0:0'),
            ('(forall (?obj - Can)\
                (not (BaxterInGripperRot ?robot ?obj))\
            )', '0:0'),
            ('(forall (?obj - Can) \
                (BaxterStationary ?obj)\
            )', '0:19'),
            ('(forall (?obj - Can) (BaxterStationaryNEq ?obj ?can))', '20:38'),
            ('(forall (?w - Obstacle)\
                (BaxterStationaryW ?w)\
            )', '0:38'),
            ('(BaxterStationaryBase ?robot)', '0:38'),
            ('(BaxterIsMP ?robot)', '0:38'),
            ('(BaxterWithinJointLimit ?robot)', '0:39'),
            # ('(forall (?w - Obstacle)\
            #     (forall (?obj - Can)\
            #         (not (BaxterCollides ?obj ?w))\
            #     )\
            # )', '0:38'),
            ('(forall (?w - Obstacle)\
                (not (BaxterRCollides ?robot ?w))\
            )', '0:39'),
            ('(forall (?obj - Can)\
                (not (BaxterObstructs ?robot ?sp ?ep ?obj))\
            )', '0:14'),
            ('(forall (?obj - Can)\
                (not (BaxterObstructsHolding ?robot ?sp ?ep ?obj ?can))\
            )', '15:39')
        ]
        self.eff = [\
            ('(not (BaxterAt ?can ?target))', '39:38') ,
            ('(not (BaxterRobotAt ?robot ?sp))', '39:38'),
            ('(BaxterRobotAt ?robot ?ep)', '39:39'),
            ('(BaxterInGripperPos ?robot ?can)', '21:39'),
            ('(BaxterInGripperRot ?robot ?can)', '21:39'),
            ('(forall (?sym1 - RobotPose)\
                (forall (?sym2 - RobotPose)\
                    (not (BaxterObstructs ?robot ?sym1 ?sym2 ?can))\
                )\
            )', '39:38'),
            ('(forall (?sym1 - Robotpose)\
                (forall (?sym2 - RobotPose)\
                    (forall (?obj - Can) (not (BaxterObstructsHolding ?robot ?sym1 ?sym2 ?can ?obj)))\
                )\
            )', '39:38')
        ]


actions = [Grasp()]
for action in actions:
    dom_str += '\n\n'
    dom_str += action.to_str()

# removes all the extra spaces
dom_str = dom_str.replace('            ', '')
dom_str = dom_str.replace('    ', '')
dom_str = dom_str.replace('    ', '')

print dom_str
f = open('baxter_grasp.domain', 'w')
f.write(dom_str)
