import unittest
from pma import pr_graph

class TestPRGraph(unittest.TestCase):
    # TODO
    def test_goal_test(self):
        d_c = {
            'Types':' Can, Target, RobotPose, Robot, Grasp,'
            'Attribute Import Paths':'RedCircle core.util_classes.circle, BlueCircle core.util_classes.circle, GreenCircle core.util_classes.circle, Vector2d core.util_classes.matrix, GridWorldViewer core.util_classes.viewer',

            'Predicates Import Path':' core.util_classes.namo_predicates',

            'Primitive Predicates':' geom, Can, RedCircle; pose, Can, Vector2d; geom, Target, BlueCircle; pose, Target, Vector2d; value, RobotPose, Vector2d; geom, Robot, GreenCircle; pose, Robot, Vector2d; value, Grasp, Vector2D;',

            'Derived Predicates':' At, Can, Target; RobotAt, Robot, RobotPose; InGripper, Robot, Can, Grasp; InContact, Robot, RobotPose, Target; NotObstructs, Robot, RobotPose, Can; NotObstructsHolding, Robot, RobotPose, Can, Can, Grasp; Stationary Can; GraspValid RobotPose Target Grasp;',

            'Action moveto 20':' (?robot - Robot ?start - RobotPose ?end - RobotPose) (forall (?c-Can ?g-Grasp) (not (InGripper ?robot ?c ?g))) (RobotAt ?robot ?start) (forall (?obj - Can ?t - Target) (or (not (At ?obj ?t)) (not (NotObstructs ?robot ?end ?obj))))) (not (RobotAt ?robot ?start)) (RobotAt ?robot ?end) 0:0 0:0 0:19 19:19 19:19',

            'Action movetoholding 20':' (?robot - Robot ?start - RobotPose ?end - RobotPose ?c - Can ?g - Grasp) (RobotAt ?robot ?start) (InGripper ?robot ?c ?g) (forall (?obj - Can) (or (not (At ?obj ?t)) (not (NotObstructsHolding ?robot ?end ?obj ?c)))) (not (RobotAt ?robot ?start)) (RobotAt ?robot ?end) 0:0 0:19 0:19 19:19 19:19',

            'Action grasp 2':' (?robot - Robot ?can - Can ?target - Target ?gp - RobotPose ?g - Grasp) (and (At ?can ?target) (RobotAt ?robot ?gp) (InContact ?robot ?gp ?target) (GraspValid ?gp ?target ?g) (forall (?obj - Can ?g - Grasp) (not (InGripper ?robot ?obj ?g)))) (and (not (At ?can ?target)) (InGripper ?robot ?can ?g) (forall (?sym - RobotPose) (not (NotObstructs ?robot ?sym ?can))) (forall (?sym-Robotpose ?obj-Can) (not (NotObstructs ?robot ?sym ?can ?obj)))) 0:0 0:0 0:0 0:0 0:0 0:1 1:1 1:1 1:1',

            'Action putdown 2':' (?robot - Robot ?can - Can ?target - Target ?pdp - RobotPose ?g - Grasp) (and (RobotAt ?robot ?pdp) (InContact ?robot ?pdp ?target) (GraspValid ?pdp ?target ?g) (InGripper ?robot ?can ?g) (forall (?obj - Can) (not (At ?obj ?target))) (forall (?obj - Can) (not (NotObstructsHolding ?robot ?pdp ?obj ?can ?g)))) (and (At ?can ?target) (not (InGripper ?robot ?can ?g))) 0:0 0:0 0:0 0:0 0:0 0:1 1:1 1:1'}
        p_c = {
            'Init':
                '(geom target0 1), \
                (pose target0 [3,5]), \
                (value pdp_target0 [3,7.05]), \
                (geom target1 1), \
                (pose target1 [3,2]), \
                (value pdp_target1 [3,4.05]), \
                (geom target2 1), \
                (pose target2 [5,3]), \
                (value pdp_target2 [5,5.05]), \
                (geom can0 1), \
                (pose can0 [3,5]), \
                (value gp_can0 [5.05,5]), \
                (geom can1 1), \
                (pose can1 [3,2]), \
                (value gp_can1 [5.05,2]), \
                (geom pr2 1), \
                (pose pr2 [0,7]), \
                (value robot_init_pose [0,7]),\
                (pose ws [0,0]), \
                (w ws 8), \
                (h ws 9), \
                (size ws 1), \
                (viewer ws); \
                (At can0 target0), \
                (InContact pr2 gp_can0 target0), \
                (At can1 target1), \
                (InContact pr2 gp_can1 target1), \
                (InContact pr2 pdp_target0 target0), \
                (InContact pr2 pdp_target1 target1), \
                (InContact pr2 pdp_target2 target2), \
                (RobotAt pr2 robot_init_pose)',
    		'Objects': 'Target (name target0); \
    			RobotPose (name pdp_target0); \
    			Can (name can0); \
    			RobotPose (name gp_can0); \
    			Target (name target1); \
    			RobotPose (name pdp_target1); \
    			Can (name can1); \
    			RobotPose (name gp_can1); \
    			Target (name target2); \
    			RobotPose (name pdp_target2); \
    			Robot (name pr2); \
    			RobotPose (name robot_init_pose); \
    			Workspace (name ws)',
    		'Goal':
                '(At can0 target0), \
                (At can1 target1)'}
        s_c = {'LLSolver': 'NAMOSolver', 'HLSolver': 'FFSolver'}
        plan, msg = pr_graph.p_mod_abs(d_c, p_c, s_c)
        self.assertFalse(plan)
        self.assertEqual(msg, "Goal is already satisfied. No planning done.")

if __name__ == '__main__':
    unittest.main()
