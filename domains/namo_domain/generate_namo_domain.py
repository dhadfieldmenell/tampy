dom_str = """
# AUTOGENERATED. DO NOT EDIT.
# Configuration file for CAN domain. Blank lines and lines beginning with # are filtered out.

# implicity, all types require a name
Types: Can, Target, RobotPose, Robot, Grasp, Obstacle

# Define the class location of each non-standard attribute type used in the above parameter type descriptions.

Attribute Import Paths: RedCircle core.util_classes.items, BlueCircle core.util_classes.items, GreenCircle core.util_classes.items, Vector1d core.util_classes.matrix, Vector2d core.util_classes.matrix, Wall core.util_classes.items

Predicates Import Path: core.util_classes.namo_predicates

"""

prim_pred_str = 'Primitive Predicates: geom, Can, RedCircle; pose, Can, Vector2d; geom, Target, BlueCircle; value, Target, Vector2d; value, RobotPose, Vector2d; gripper, RobotPose, Vector1d; geom, RobotPose, BlueCircle; pose, Robot, Vector2d; gripper, Robot, Vector1d; value, Grasp, Vector2d; geom, Obstacle, Wall; pose, Obstacle, Vector2d; vel, RobotPose, Vector1d; acc, RobotPose, Vector1d; vel, Robot, Vector1d; acc, Robot, Vector1d; geom, Robot, GreenCircle'
dom_str += prim_pred_str + '\n\n'

der_pred_str = 'Derived Predicates: At, Can, Target; RobotAt, Robot, RobotPose; InGripper, Robot, Can, Grasp; Approach, Robot, Can, Grasp; ApproachTarget, Robot, Target, Grasp; Obstructs, Robot, Can, Can, Can; ObstructsHolding, Robot, Target, Target, Can, Can; WideObstructsHolding, Robot, Target, Target, Can, Can; Stationary, Can; RobotStationary, Robot; StationaryNEq, Can, Can; IsMP, Robot; StationaryW, Obstacle; Collides, Can, Obstacle; CanCollides, Can, Can; RCollides, Robot, Obstacle; GripperClosed, Robot; Near, Can, Target;  RobotAtGrasp, Robot, Can, Grasp; RobotWithinReach, Robot, Target; RobotNearGrasp, Robot, Can, Grasp; RobotWithinBounds, Robot; WideObstructs, Robot, Can, Can, Can; AtNEq, Can, Can, Target; PoseCollides, RobotPose, Obstacle; TargetCollides, Target, Obstacle; TargetGraspCollides, Target, Obstacle, Grasp; TargetCanGraspCollides, Target, Can, Grasp, Can; CanGraspCollides, Can, Obstacle, Grasp; HLPoseUsed, RobotPose; HLAtGrasp, Robot, Can, Grasp; RobotPoseAtGrasp, RobotPose, Target, Grasp; HLPoseAtGrasp, RobotPose, Target, Grasp'
dom_str += der_pred_str + '\n'

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

class MoveTo(Action):
    def __init__(self):
        self.name = 'moveto'
        self.timesteps = 25
        et = self.timesteps - 1
        self.args = '(?robot - Robot ?can - Can ?target - Target ?sp - RobotPose ?gp - RobotPose ?g - Grasp ?end - Target)' 
        self.pre = [\
                ('(At ?can ?target)', '0:0'),
                ('(forall (?gr - Grasp) (not (RobotAtGrasp ?robot ?can ?gr)))', '0:0'),
                ('(forall (?w - Obstacle) (not (CanGraspCollides ?can ?w ?g)))', '0:0'),
                ('(not (GripperClosed ?robot))', '1:{0}'.format(et-1)),
                ('(forall (?obj - Can) (Stationary ?obj))', '0:{0}'.format(et-1)),
                ('(forall (?w - Obstacle) (StationaryW ?w))', '0:{0}'.format(et-1)),
                ('(IsMP ?robot)', '0:{0}'.format(et-1)),
                #('(Approach ?robot ?can ?g)', '{0}:{0}'.format(et-1)),
                ('(forall (?w - Obstacle) (forall (?obj - Can) (not (Collides ?obj ?w))))', '0:{0}'.format(et-1)),
                ('(forall (?w - Obstacle) (not (RCollides ?robot ?w)))', '0:{0}'.format(et-1)),
                ('(forall (?obj - Can) (not (Obstructs ?robot ?can ?can ?obj)))', '0:-1'),
                ('(forall (?obj - Can) (not (WideObstructs ?robot ?can ?can ?obj)))', '1:{0}'.format(et-2)),
                ('(forall (?obj - Can) (not (Obstructs ?robot ?can ?can ?obj)))', '1:{0}'.format(et-2)),
                ('(forall (?obj - Can) (not (Obstructs ?robot ?can ?can ?obj)))', '{0}:{1}'.format(et-1, et-1)),
                # ('(forall (?obj - Can) (not (ObstructsHolding ?robot ?can ?can ?obj ?can)))', '{0}:{1}'.format(et-2, et-1)),
        ]
        self.eff = [\
                ('(RobotAtGrasp ?robot ?can ?g)', '{0}:{0}'.format(et)),
                ('(forall (?obj - Can / ?can) (forall (?gr - Grasp) (not (RobotAtGrasp ?robot ?obj ?gr))))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (forall (?gr - Grasp / ?g) (not (RobotAtGrasp ?robot ?obj ?gr))))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (Stationary ?obj))', '{0}:{1}'.format(et, et-1)),
                #('(RobotStationary ?robot)', '{0}:{0}'.format(et-1)),
                ('(RobotStationary ?robot)', '{0}:{1}'.format(et, et-1)),
        ]

class Transfer(Action):
    def __init__(self):
        self.name = 'transfer'
        self.timesteps = 25
        et = self.timesteps - 1
        self.args = '(?robot - Robot ?start - RobotPose ?end - RobotPose ?c - Can ?t - Target ?g - Grasp ?init - Target)'
        self.pre = [\
                ('(At ?c ?init)', '0:0'),
                # ('(At ?c ?init)', '1:1'),
                ('(RobotStationary ?robot)', '0:0'),
                ('(forall (?obj - Can) (not (TargetCanGraspCollides ?t ?obj ?g ?c)))', '0:0'),
                ('(forall (?w - Obstacle) (not (TargetGraspCollides ?t ?w ?g)))', '0:0'),
                ('(forall (?obj - Can) (not (Near ?obj ?t)))', '0:0'),
                # ('(not (GripperClosed ?robot))', '0:0'),
                ('(GripperClosed ?robot)', '1:{0}'.format(et-1)),
                ('(InGripper ?robot ?c ?g)', '{0}:{0}'.format(1, et-1)),
                ('(RobotAtGrasp ?robot ?c ?g)', '{0}:{0}'.format(0)),
                #('(ApproachTarget ?robot ?t ?g)', '{0}:{0}'.format(et-2)),
                ('(forall (?obj - Can) (not (ObstructsHolding ?robot ?init ?t ?obj ?c)))', '0:{0}'.format(0)),
                ('(forall (?obj - Can) (not (WideObstructsHolding ?robot ?init ?t ?obj ?c)))', '1:{0}'.format(et-1)),
                ('(forall (?obj - Can) (not (ObstructsHolding ?robot ?init ?t ?obj ?c)))', '1:{0}'.format(et-2)),
                ('(forall (?obj - Can) (not (ObstructsHolding ?robot ?init ?t ?obj ?c)))', '{0}:{0}'.format(et-1)),
                ('(forall (?obj - Can ) (not (Obstructs ?robot ?c ?c ?obj)))', '{0}:{0}'.format(et-1)),
                ('(forall (?obj - Can) (StationaryNEq ?obj ?c))', '0:{0}'.format(et-1)), 
                ('(forall (?w - Obstacle) (StationaryW ?w))', '0:{0}'.format(et-1)), 
                ('(IsMP ?robot)', '0:{0}'.format(et-1)),
                ('(forall (?w - Obstacle) (forall (?obj - Can) (not (Collides ?obj ?w))))', '0:{0}'.format(et-1)),
                ('(forall (?w - Obstacle) (not (RCollides ?robot ?w)))', '0:{0}'.format(et-1)),
                ('(RobotStationary ?robot)', '{0}:{0}'.format(et-1)),
                ]
        self.eff = [\
                ('(At ?c ?t)', '{0}:{1}'.format(et-1, et)),
                ('(Near ?c ?t)', '{0}:{0}'.format(et)),
                ('(forall (?obj - Can) (forall (?targ - Target) (not (TargetCanGraspCollides ?targ ?c ?g ?obj))))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?targ - Target / ?t) (not (Near ?c ?targ)))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?targ - Target / ?t) (not (At ?c ?targ)))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (not (Obstructs ?robot ?obj ?obj ?c)))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (not (WideObstructs ?robot ?obj ?obj ?c)))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (not (ObstructsHolding ?robot ?init ?t ?c ?obj)))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (not (WideObstructsHolding ?robot ?init ?t ?c ?obj)))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?obj - Can) (Stationary ?obj))', '{0}:{1}'.format(et, et-1)),
                ('(forall (?gr - Grasp) (forall (?obj - Can) (not (RobotAtGrasp ?robot ?obj ?g))))', '{0}:{1}'.format(et, et-1)),
        ]

actions = [MoveTo(), Transfer()]
for action in actions:
    dom_str += '\n\n'
    dom_str += action.to_str()

# removes all the extra spaces
dom_str = dom_str.replace('            ', '')
dom_str = dom_str.replace('    ', '')
dom_str = dom_str.replace('    ', '')

print(dom_str)
f = open('namo_current.domain', 'w')
f.write(dom_str)
