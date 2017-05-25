#!/bin/bash

# nose2 test.test_core.test_util_classes.test_robot_predicates

nose2 test.test_core.test_util_classes.test_baxter_predicates.TestBaxterPredicates.test_basket_ee_reachable --debug

nose2 test.test_core.test_util_classes.test_baxter_predicates.TestBaxterPredicates.test_ee_reachable --debug

nose2 test.test_core.test_util_classes.test_baxter_predicates.TestBaxterPredicates.test_basket_in_gripper --debug

nose2 test.test_core.test_util_classes.test_baxter_predicates.TestBaxterPredicates.test_in_gripper --debug

nose2 test.test_core.test_util_classes.test_baxter_predicates.TestBaxterPredicates.test_eereachable_inv --debug


nose2 test.test_core.test_util_classes.test_pr2_predicates.TestPR2Predicates.test_in_gripper --debug

nose2 test.test_core.test_util_classes.test_pr2_predicates.TestPR2Predicates.test_ee_reachable --debug


# nose2 test.test_core.test_util_classes.test_baxter.TestBaxter.test_baxter_ik

# nose2 test.test_core.test_util_classes.test_baxter.TestBaxter.test_can_world --debug

# nose2 test.test_core.test_util_classes.test_baxter.TestBaxter.test_rrt_planner

# nose2 test.test_core.test_util_classes.test_baxter.TestBaxter.test_move_holding_env --debug


# nose2 test.test_pma.test_robot_ll_solver.TestRobotLLSolver --debug

# nose2 test.test_core.test_util_classes.test_baxter_sampling.TestBaxterSampling.test_resampling_rrt --debug
